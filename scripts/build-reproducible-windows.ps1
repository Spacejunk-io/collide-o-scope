param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$TargetDir,
    [Parameter(Mandatory = $true)][string]$FfmpegDir,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{40}$')][string]$GitSha,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9]+$')][string]$SourceDateEpoch
)

$ErrorActionPreference = "Stop"

if (-not ("CollideReproducibleNativePaths" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public static class CollideReproducibleNativePaths
{
    private delegate uint PathExpander(string source, StringBuilder destination, uint capacity);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern uint GetLongPathNameW(
        string shortPath,
        StringBuilder longPath,
        uint capacity);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern uint GetShortPathNameW(
        string longPath,
        StringBuilder shortPath,
        uint capacity);

    private static string Expand(string path, PathExpander expander)
    {
        uint capacity = 260;
        while (true)
        {
            var result = new StringBuilder((int)capacity);
            uint length = expander(path, result, capacity);
            if (length == 0)
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            if (length < capacity)
            {
                return result.ToString();
            }
            capacity = checked(length + 1);
        }
    }

    public static string GetLongPath(string path)
    {
        return Expand(path, GetLongPathNameW);
    }

    public static string GetShortPath(string path)
    {
        return Expand(path, GetShortPathNameW);
    }
}
'@
}

$resolvedSource = (Resolve-Path -LiteralPath $SourceRoot).Path
$source = [CollideReproducibleNativePaths]::GetLongPath($resolvedSource)
$sourceShort = [CollideReproducibleNativePaths]::GetShortPath($source)
$resolvedFfmpeg = (Resolve-Path -LiteralPath $FfmpegDir).Path
$ffmpeg = [CollideReproducibleNativePaths]::GetLongPath($resolvedFfmpeg)
$ffmpegShort = [CollideReproducibleNativePaths]::GetShortPath($ffmpeg)
$target = [System.IO.Path]::GetFullPath($TargetDir)
$userProfileCandidate = [Environment]::GetEnvironmentVariable("USERPROFILE", "Process")
if (
    [string]::IsNullOrWhiteSpace($userProfileCandidate) -or
    -not (Test-Path -LiteralPath $userProfileCandidate -PathType Container)
) {
    throw "USERPROFILE must name an existing directory"
}
$resolvedUserProfile = (Resolve-Path -LiteralPath $userProfileCandidate).Path
$userProfile = [CollideReproducibleNativePaths]::GetLongPath($resolvedUserProfile)
$userProfileShort = [CollideReproducibleNativePaths]::GetShortPath($userProfile)
$cargoHomeOverride = [Environment]::GetEnvironmentVariable("CARGO_HOME", "Process")
if ([string]::IsNullOrEmpty($cargoHomeOverride)) {
    $cargoHomeCandidate = Join-Path $userProfile ".cargo"
} else {
    if ([string]::IsNullOrWhiteSpace($cargoHomeOverride)) {
        throw "CARGO_HOME must not be whitespace"
    }
    $cargoHomeCandidate = $cargoHomeOverride
}
if (-not (Test-Path -LiteralPath $cargoHomeCandidate -PathType Container)) {
    throw "Cargo home directory is missing: $cargoHomeCandidate"
}
$resolvedCargoHome = (Resolve-Path -LiteralPath $cargoHomeCandidate).Path
$cargoHome = [CollideReproducibleNativePaths]::GetLongPath($resolvedCargoHome)
$cargoHomeShort = [CollideReproducibleNativePaths]::GetShortPath($cargoHome)
foreach ($cargoHomePath in @($cargoHome, $cargoHomeShort)) {
    if (
        -not [System.IO.Path]::IsPathRooted($cargoHomePath) -or
        -not (Test-Path -LiteralPath $cargoHomePath -PathType Container)
    ) {
        throw "Resolved Cargo home is not an existing absolute directory: $cargoHomePath"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $source ".git"))) {
    throw "SourceRoot is not a Git checkout: $source"
}
if (-not (Test-Path -LiteralPath (Join-Path $ffmpeg "bin"))) {
    throw "FFmpeg bin directory is missing: $ffmpeg"
}
$dirty = @(git -C $source status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0 -or $dirty.Count -ne 0) {
    throw "Reproducible builds require an entirely clean source checkout"
}
$actualSha = (git -C $source rev-parse HEAD).Trim().ToLowerInvariant()
if ($LASTEXITCODE -ne 0 -or $actualSha -ne $GitSha.ToLowerInvariant()) {
    throw "Checkout SHA $actualSha does not match requested $GitSha"
}
if (Test-Path -LiteralPath $target) {
    if (@(Get-ChildItem -LiteralPath $target -Force).Count -ne 0) {
        throw "TargetDir must be absent or empty: $target"
    }
} else {
    New-Item -ItemType Directory -Path $target | Out-Null
}
$resolvedTarget = (Resolve-Path -LiteralPath $target).Path
$target = [CollideReproducibleNativePaths]::GetLongPath($resolvedTarget)
$targetShort = [CollideReproducibleNativePaths]::GetShortPath($target)

if (-not [string]::IsNullOrEmpty(
    [Environment]::GetEnvironmentVariable("CARGO_BUILD_TARGET", "Process")
)) {
    throw "CARGO_BUILD_TARGET is not permitted for the reproducible Windows build"
}
$rustcVersion = (rustc -vV) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "rustc -vV failed while resolving the native target"
}
$rustHostMatch = [regex]::Match($rustcVersion, '(?m)^host: ([A-Za-z0-9_.-]+)$')
if (-not $rustHostMatch.Success) {
    throw "rustc -vV did not report exactly one canonical host target"
}
$nativeTarget = $rustHostMatch.Groups[1].Value
$nativeTargetUnderscored = $nativeTarget.Replace('-', '_').Replace('.', '_')
$higherPriorityNativeFlagNames = @(
    "HOST_CFLAGS", "TARGET_CFLAGS",
    "CFLAGS_$nativeTarget", "CFLAGS_$nativeTargetUnderscored",
    "HOST_CFLAGS_$nativeTargetUnderscored", "TARGET_CFLAGS_$nativeTargetUnderscored",
    "HOST_CXXFLAGS", "TARGET_CXXFLAGS",
    "CXXFLAGS_$nativeTarget", "CXXFLAGS_$nativeTargetUnderscored",
    "HOST_CXXFLAGS_$nativeTargetUnderscored", "TARGET_CXXFLAGS_$nativeTargetUnderscored",
    "AWS_LC_SYS_CFLAGS", "AWS_LC_SYS_CFLAGS_$nativeTargetUnderscored",
    "AWS_LC_SYS_HOST_CFLAGS", "AWS_LC_SYS_HOST_CFLAGS_$nativeTargetUnderscored",
    "AWS_LC_SYS_TARGET_CFLAGS", "AWS_LC_SYS_TARGET_CFLAGS_$nativeTargetUnderscored",
    "AWS_LC_SYS_CXXFLAGS", "AWS_LC_SYS_CXXFLAGS_$nativeTargetUnderscored",
    "AWS_LC_SYS_HOST_CXXFLAGS", "AWS_LC_SYS_HOST_CXXFLAGS_$nativeTargetUnderscored",
    "AWS_LC_SYS_TARGET_CXXFLAGS", "AWS_LC_SYS_TARGET_CXXFLAGS_$nativeTargetUnderscored"
)
foreach ($nativeFlagName in $higherPriorityNativeFlagNames) {
    $nativeFlagValue = [Environment]::GetEnvironmentVariable($nativeFlagName, "Process")
    if ($null -ne $nativeFlagValue) {
        throw "higher-priority native compiler flags are not permitted: $nativeFlagName"
    }
}
$cmakeEnvironmentNames = @(
    foreach ($cmakeVariable in @("CMAKE_GENERATOR", "CMAKE_TOOLCHAIN_FILE")) {
        $cmakeVariable
        "${cmakeVariable}_$nativeTarget"
        "${cmakeVariable}_$nativeTargetUnderscored"
        "HOST_$cmakeVariable"
        "AWS_LC_SYS_$cmakeVariable"
        "AWS_LC_SYS_${cmakeVariable}_$nativeTargetUnderscored"
    }
) | Select-Object -Unique
foreach ($cmakeEnvironmentName in $cmakeEnvironmentNames) {
    $cmakeEnvironmentValue = [Environment]::GetEnvironmentVariable($cmakeEnvironmentName, "Process")
    if ($null -ne $cmakeEnvironmentValue) {
        throw "ambient CMake configuration is not permitted: $cmakeEnvironmentName"
    }
}

$saved = @{}
$names = @(
    "CARGO_HOME", "CARGO_ENCODED_RUSTFLAGS", "CARGO_TARGET_DIR", "COLLIDE_BUILD_GIT_SHA",
    "COLLIDE_BUILD_GIT_DIRTY", "COLLIDE_PUBLISHED_ARTIFACT", "FFMPEG_DIR",
    "SOURCE_DATE_EPOCH", "CC_SHELL_ESCAPED_FLAGS", "CFLAGS", "CXXFLAGS", "CL", "_CL_", "PATH"
)
foreach ($name in $names) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    $env:CARGO_HOME = $cargoHome
    $installedCargoTools = (cargo install --list) -join "`n"
    if (
        $LASTEXITCODE -ne 0 -or
        $installedCargoTools -notmatch '(?m)^cargo-auditable v0\.7\.5:$'
    ) {
        throw "cargo-auditable 0.7.5 is required"
    }

    $unitSeparator = [char]0x1f
    $remappedSource = $source.Replace('\', '/')
    $remappedTarget = $target.Replace('\', '/')
    $remappedCargoHome = $cargoHome.Replace('\', '/')
    $remappedFfmpeg = $ffmpeg.Replace('\', '/')
    $remappedFfmpegShort = $ffmpegShort.Replace('\', '/')
    $encodedFlags = @(
        "-C", "link-arg=/Brepro",
        "--remap-path-prefix=$remappedSource=/collide-o-scope",
        "--remap-path-prefix=$remappedTarget=/collide-o-scope-target",
        "--remap-path-prefix=$remappedCargoHome=/cargo-home",
        "--remap-path-prefix=$remappedFfmpeg=/ffmpeg",
        "--remap-path-prefix=$remappedFfmpegShort=/ffmpeg"
    ) -join $unitSeparator
    $env:CARGO_ENCODED_RUSTFLAGS = $encodedFlags
    $env:CARGO_TARGET_DIR = $target
    $env:COLLIDE_BUILD_GIT_SHA = $GitSha.ToLowerInvariant()
    $env:COLLIDE_BUILD_GIT_DIRTY = "false"
    $env:COLLIDE_PUBLISHED_ARTIFACT = "true"
    $env:FFMPEG_DIR = $ffmpeg
    $env:SOURCE_DATE_EPOCH = $SourceDateEpoch
    $nativeTrimSource = "/d1trimfile:$source"
    $nativeTrimSourceShort = "/d1trimfile:$sourceShort"
    $nativeTrimTarget = "/d1trimfile:$target"
    $nativeTrimTargetShort = "/d1trimfile:$targetShort"
    $nativeTrimLong = "/d1trimfile:$cargoHome"
    $nativeTrimShort = "/d1trimfile:$cargoHomeShort"
    $nativeTrimFfmpeg = "/d1trimfile:$ffmpeg"
    $nativeTrimFfmpegShort = "/d1trimfile:$ffmpegShort"
    $nativeTrimArguments = @(
        $nativeTrimSource,
        $nativeTrimSourceShort,
        $nativeTrimTarget,
        $nativeTrimTargetShort,
        $nativeTrimLong,
        $nativeTrimShort,
        $nativeTrimFfmpeg,
        $nativeTrimFfmpegShort
    )
    $nativeTrimFlags = ($nativeTrimArguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $env:CC_SHELL_ESCAPED_FLAGS = "1"
    $env:CFLAGS = $nativeTrimFlags
    $env:CXXFLAGS = $nativeTrimFlags
    [Environment]::SetEnvironmentVariable("CL", $null, "Process")
    [Environment]::SetEnvironmentVariable("_CL_", $null, "Process")
    $env:PATH = (Join-Path $ffmpeg "bin") + ";" + $env:PATH

    Push-Location $source
    try {
        cargo auditable build --locked --release --bin collide-o-scope
        if ($LASTEXITCODE -ne 0) { throw "cargo auditable build failed" }
    } finally {
        Pop-Location
    }

    $executable = Join-Path $target "release\collide-o-scope.exe"
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Expected release executable is missing: $executable"
    }
    $executableBytes = [System.IO.File]::ReadAllBytes($executable)
    $latin1 = [Text.Encoding]::GetEncoding(28591)
    $binaryView = $latin1.GetString($executableBytes)
    $needleEncodings = @(
        [Text.Encoding]::UTF8,
        [Text.Encoding]::Unicode,
        [Text.Encoding]::BigEndianUnicode
    )
    $profilesRoot = (Split-Path -Parent $userProfile).TrimEnd([char[]]@('\', '/')) + '\'
    $builderSpecificPaths = @(
        $source,
        $sourceShort,
        $target,
        $targetShort,
        $cargoHome,
        $cargoHomeShort,
        $ffmpeg,
        $ffmpegShort,
        $userProfile,
        $userProfileShort,
        $profilesRoot,
        'C:\Users\'
    ) | Select-Object -Unique
    foreach ($builderSpecificPath in $builderSpecificPaths) {
        $spellings = @(
            $builderSpecificPath,
            $builderSpecificPath.Replace('\', '/')
        ) | Select-Object -Unique
        foreach ($spelling in $spellings) {
            foreach ($encoding in $needleEncodings) {
                $needleView = $latin1.GetString($encoding.GetBytes($spelling))
                if ($binaryView.IndexOf($needleView, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                    throw "release executable contains a builder-specific path"
                }
            }
        }
    }
    $identityJson = & $executable --version-json
    if ($LASTEXITCODE -ne 0) { throw "built executable rejected --version-json" }
    $identity = $identityJson | ConvertFrom-Json
    if ($identity.git_dirty -or -not $identity.published_artifact) {
        throw "release executable has a dirty or unpublished BuildIdentity"
    }
    if ($identity.git_sha -ne $GitSha.ToLowerInvariant()) {
        throw "release executable embeds the wrong Git SHA"
    }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $executable).Hash.ToLowerInvariant()
    [pscustomobject]@{
        executable = $executable
        sha256 = $hash
        identity_sha256 = $identity.identity_sha256
    } | ConvertTo-Json -Compress
} finally {
    foreach ($name in $names) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], "Process")
    }
}
