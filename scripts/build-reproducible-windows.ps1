param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$TargetDir,
    [Parameter(Mandatory = $true)][string]$FfmpegDir,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{40}$')][string]$GitSha,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9]+$')][string]$SourceDateEpoch
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

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

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateDirectoryW(string path, IntPtr securityAttributes);

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

    public static void CreateNewDirectory(string path)
    {
        if (!CreateDirectoryW(path, IntPtr.Zero))
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
    }
}
'@
}

function Get-NormalizedFullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path).TrimEnd([char[]]@('\', '/'))
}

function Test-PathWithin {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Boundary
    )
    $candidatePath = (Get-NormalizedFullPath $Candidate) + '\'
    $boundaryPath = (Get-NormalizedFullPath $Boundary) + '\'
    return $candidatePath.StartsWith($boundaryPath, [StringComparison]::OrdinalIgnoreCase)
}

function Assert-PathsDisjoint {
    param(
        [Parameter(Mandatory = $true)][string]$Left,
        [Parameter(Mandatory = $true)][string]$Right,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if ((Test-PathWithin $Left $Right) -or (Test-PathWithin $Right $Left)) {
        throw "$Description paths overlap"
    }
}

function Assert-NoReparsePoints {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$Recurse
    )
    $root = Get-Item -LiteralPath $Path -Force
    $items = @($root)
    if ($Recurse -and $root.PSIsContainer) {
        $items += @(Get-ChildItem -LiteralPath $root.FullName -Force -Recurse)
    }
    foreach ($item in $items) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "reparse points are not permitted: $($item.FullName)"
        }
    }
}

function Assert-OnlyDefaultDataStream {
    param([Parameter(Mandatory = $true)][string]$Path)
    $streams = @(Get-Item -LiteralPath $Path -Stream *)
    if ($streams.Count -ne 1 -or $streams[0].Stream -cne ':$DATA') {
        throw "alternate data streams are not permitted: $Path"
    }
}

function Get-TreeManifestDigest {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string[]]$ExcludedRelativePaths = @()
    )
    $root = Get-NormalizedFullPath $Path
    $excluded = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($excludedPath in $ExcludedRelativePaths) {
        $normalizedExcludedPath = $excludedPath.TrimStart([char[]]@('\', '/')).Replace('\', '/')
        if (-not $excluded.Add($normalizedExcludedPath)) {
            throw "duplicate tree-manifest exclusion: $normalizedExcludedPath"
        }
    }
    $records = [Collections.Generic.List[string]]::new()
    foreach ($file in @(Get-ChildItem -LiteralPath $root -Force -Recurse -File)) {
        $relative = $file.FullName.Substring($root.Length).TrimStart([char[]]@('\', '/')).Replace('\', '/')
        if ($excluded.Contains($relative)) {
            continue
        }
        Assert-OnlyDefaultDataStream -Path $file.FullName
        $digest = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $records.Add("$relative`0$($file.Length)`0$digest")
    }
    $records.Sort([StringComparer]::Ordinal)
    $bytes = [Text.Encoding]::UTF8.GetBytes([string]::Join("`n", $records))
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
}

function Assert-NoReparsePathChain {
    param([Parameter(Mandatory = $true)][string]$Path)
    $fullPath = [IO.Path]::GetFullPath($Path)
    $pathRoot = [IO.Path]::GetPathRoot($fullPath)
    if ([string]::IsNullOrWhiteSpace($pathRoot)) {
        throw "path has no rooted filesystem boundary: $Path"
    }
    $current = $pathRoot
    $relative = $fullPath.Substring($pathRoot.Length)
    foreach ($component in $relative.Split([char[]]@('\', '/'), [StringSplitOptions]::RemoveEmptyEntries)) {
        $current = Join-Path $current $component
        if (-not (Test-Path -LiteralPath $current)) {
            throw "reparse-path validation requires an existing path component: $current"
        }
        $item = Get-Item -LiteralPath $current -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "reparse-point path ancestors are not permitted: $current"
        }
    }
}

function Copy-VerifiedTree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "copy source directory is missing: $Source"
    }
    if (Test-Path -LiteralPath $Destination) {
        throw "copy destination must be absent: $Destination"
    }
    Assert-NoReparsePoints -Path $Source -Recurse
    $before = Get-TreeManifestDigest $Source
    & robocopy.exe $Source $Destination /E /COPY:DAT /DCOPY:DAT /XJ /R:2 /W:1 /NFL /NDL /NJH /NJS /NP
    $robocopyExit = $LASTEXITCODE
    if ($robocopyExit -ge 8) {
        throw "verified tree copy failed with robocopy exit code $robocopyExit"
    }
    Assert-NoReparsePoints -Path $Destination -Recurse
    $after = Get-TreeManifestDigest $Source
    $copied = Get-TreeManifestDigest $Destination
    if ($before -cne $after -or $before -cne $copied) {
        throw "tree changed during verified copy or destination bytes disagree"
    }
    return $copied
}

function Copy-VerifiedFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "copy source file is missing: $Source"
    }
    $sourceItem = Get-Item -LiteralPath $Source -Force
    if (($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "reparse-point files are not permitted: $Source"
    }
    Assert-OnlyDefaultDataStream -Path $Source
    if (Test-Path -LiteralPath $Destination) {
        throw "copy destination must be absent: $Destination"
    }
    $before = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.File]::Copy($Source, $Destination, $false)
    $after = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant()
    $copied = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    Assert-OnlyDefaultDataStream -Path $Destination
    if ($before -cne $after -or $before -cne $copied) {
        throw "file changed during verified copy or destination bytes disagree"
    }
    return $copied
}

function Assert-NoCargoConfiguration {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$CargoHomePath
    )
    $candidates = [Collections.Generic.List[string]]::new()
    $current = [IO.DirectoryInfo]::new((Get-NormalizedFullPath $SourcePath))
    while ($null -ne $current) {
        $candidates.Add((Join-Path $current.FullName '.cargo\config'))
        $candidates.Add((Join-Path $current.FullName '.cargo\config.toml'))
        $current = $current.Parent
    }
    foreach ($name in @('config', 'config.toml', 'credentials', 'credentials.toml')) {
        $candidates.Add((Join-Path $CargoHomePath $name))
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            throw "ambient Cargo configuration or credentials are not permitted: $candidate"
        }
    }
}

function Set-ProcessEnvironmentValue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [AllowNull()][string]$Value
    )
    [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
    $observed = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ($null -eq $Value) {
        if ($null -ne $observed) {
            throw "failed to clear process environment variable $Name"
        }
    } elseif ($observed -cne $Value) {
        throw "failed to set process environment variable $Name"
    }
}

function Assert-PortableExecutableHasNoCodeView {
    param([Parameter(Mandatory = $true)][string]$Executable)
    $stream = [IO.File]::Open($Executable, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try {
        $reader = [Reflection.PortableExecutable.PEReader]::new($stream)
        try {
            foreach ($entry in $reader.ReadDebugDirectory()) {
                if ($entry.Type -eq [Reflection.PortableExecutable.DebugDirectoryEntryType]::CodeView) {
                    throw "release executable contains a CodeView/PDB debug record"
                }
            }
        } finally {
            $reader.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

$canonicalRoot = 'C:\cosrepro'
$canonicalMutexName = 'Local\CollideOScope.Repro.Stage.v1'
$canonicalSource = Join-Path $canonicalRoot 'src'
$canonicalCargoHome = Join-Path $canonicalRoot 'cargo'
$canonicalFfmpeg = Join-Path $canonicalRoot 'ffmpeg'
$canonicalTarget = Join-Path $canonicalRoot 'target'
$canonicalTemp = Join-Path $canonicalRoot 'tmp'
$ownerPath = Join-Path $canonicalRoot 'owner'
$sourceArchive = Join-Path $canonicalRoot 'src.tar'

$drive = [IO.DriveInfo]::new('C:\')
if (-not $drive.IsReady -or $drive.DriveType -ne [IO.DriveType]::Fixed -or $drive.DriveFormat -cne 'NTFS') {
    throw 'canonical reproducible staging requires the fixed NTFS C: volume'
}
Assert-NoReparsePoints -Path 'C:\'

$resolvedSource = (Resolve-Path -LiteralPath $SourceRoot).Path
$inputSource = [CollideReproducibleNativePaths]::GetLongPath($resolvedSource)
$inputSourceShort = [CollideReproducibleNativePaths]::GetShortPath($inputSource)
$resolvedFfmpeg = (Resolve-Path -LiteralPath $FfmpegDir).Path
$inputFfmpeg = [CollideReproducibleNativePaths]::GetLongPath($resolvedFfmpeg)
$inputFfmpegShort = [CollideReproducibleNativePaths]::GetShortPath($inputFfmpeg)
$requestedOutputTarget = Get-NormalizedFullPath $TargetDir
$requestedOutputParent = Split-Path -Parent $requestedOutputTarget
if ([string]::IsNullOrWhiteSpace($requestedOutputParent) -or -not (Test-Path -LiteralPath $requestedOutputParent -PathType Container)) {
    throw "TargetDir parent must already exist: $requestedOutputParent"
}
$outputParent = [CollideReproducibleNativePaths]::GetLongPath((Resolve-Path -LiteralPath $requestedOutputParent).Path)
$outputTarget = Join-Path $outputParent (Split-Path -Leaf $requestedOutputTarget)
if (Test-Path -LiteralPath $outputTarget) {
    throw "TargetDir must be absent: $outputTarget"
}

$userProfileCandidate = [Environment]::GetEnvironmentVariable('USERPROFILE', 'Process')
if (
    [string]::IsNullOrWhiteSpace($userProfileCandidate) -or
    -not (Test-Path -LiteralPath $userProfileCandidate -PathType Container)
) {
    throw 'USERPROFILE must name an existing directory'
}
$resolvedUserProfile = (Resolve-Path -LiteralPath $userProfileCandidate).Path
$userProfile = [CollideReproducibleNativePaths]::GetLongPath($resolvedUserProfile)
$userProfileShort = [CollideReproducibleNativePaths]::GetShortPath($userProfile)
$cargoHomeOverride = [Environment]::GetEnvironmentVariable('CARGO_HOME', 'Process')
if ([string]::IsNullOrEmpty($cargoHomeOverride)) {
    $cargoSeedCandidate = Join-Path $userProfile '.cargo'
} else {
    if ([string]::IsNullOrWhiteSpace($cargoHomeOverride)) {
        throw 'CARGO_HOME must not be whitespace'
    }
    $cargoSeedCandidate = $cargoHomeOverride
}
if (-not (Test-Path -LiteralPath $cargoSeedCandidate -PathType Container)) {
    throw "Cargo seed directory is missing: $cargoSeedCandidate"
}
$resolvedCargoSeed = (Resolve-Path -LiteralPath $cargoSeedCandidate).Path
$inputCargoSeed = [CollideReproducibleNativePaths]::GetLongPath($resolvedCargoSeed)
$inputCargoSeedShort = [CollideReproducibleNativePaths]::GetShortPath($inputCargoSeed)

foreach ($requiredDirectory in @(
    $inputSource,
    (Join-Path $inputFfmpeg 'bin'),
    (Join-Path $inputFfmpeg 'include'),
    (Join-Path $inputFfmpeg 'lib'),
    (Join-Path $inputCargoSeed 'registry\cache'),
    (Join-Path $inputCargoSeed 'registry\index'),
    (Join-Path $inputCargoSeed 'git\db')
)) {
    if (-not (Test-Path -LiteralPath $requiredDirectory -PathType Container)) {
        throw "required reproducible-build input is missing: $requiredDirectory"
    }
    Assert-NoReparsePathChain -Path $requiredDirectory
}
foreach ($requiredFile in @(
    (Join-Path $inputCargoSeed 'bin\cargo-auditable.exe'),
    (Join-Path $inputCargoSeed '.crates.toml'),
    (Join-Path $inputCargoSeed '.crates2.json'),
    (Join-Path $inputFfmpeg 'bin\ffmpeg.exe'),
    (Join-Path $inputFfmpeg 'bin\ffprobe.exe')
)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "required Cargo seed file is missing: $requiredFile"
    }
    Assert-NoReparsePathChain -Path $requiredFile
}
$inputFfmpegBin = [CollideReproducibleNativePaths]::GetLongPath((Join-Path $inputFfmpeg 'bin'))
$inputFfmpegBinShort = [CollideReproducibleNativePaths]::GetShortPath($inputFfmpegBin)
$ambientPath = [Environment]::GetEnvironmentVariable('PATH', 'Process')
if ([string]::IsNullOrWhiteSpace($ambientPath)) {
    throw 'PATH must expose the reviewed Git, Rust, LLVM, and MSVC tools'
}
foreach ($pathEntry in $ambientPath.Split([char[]]@(';'), [StringSplitOptions]::None)) {
    $trimmedPathEntry = $pathEntry.Trim().Trim([char[]]@('"'))
    if ([string]::IsNullOrWhiteSpace($trimmedPathEntry)) {
        throw 'PATH contains an empty entry whose lookup meaning depends on the working directory'
    }
    if (-not [IO.Path]::IsPathFullyQualified($trimmedPathEntry)) {
        throw "PATH contains a relative entry: $trimmedPathEntry"
    }
    try {
        $normalizedPathEntry = Get-NormalizedFullPath $trimmedPathEntry
    } catch {
        throw "PATH contains an invalid entry: $trimmedPathEntry"
    }
    if (
        $normalizedPathEntry -ieq (Get-NormalizedFullPath $inputFfmpegBin) -or
        $normalizedPathEntry -ieq (Get-NormalizedFullPath $inputFfmpegBinShort)
    ) {
        throw 'caller FFmpeg binaries must not be on PATH during the reproducible build'
    }
}

Assert-NoReparsePathChain -Path $inputSource
Assert-NoReparsePathChain -Path $inputFfmpeg
Assert-NoReparsePathChain -Path $inputCargoSeed
Assert-NoReparsePathChain -Path $outputParent
Assert-NoReparsePoints -Path $inputSource -Recurse
Assert-NoReparsePoints -Path $inputFfmpeg -Recurse
Assert-NoReparsePoints -Path $inputCargoSeed
Assert-NoReparsePoints -Path $outputParent
Assert-NoCargoConfiguration -SourcePath $inputSource -CargoHomePath $inputCargoSeed
foreach ($inputPath in @($inputSource, $inputCargoSeed, $inputFfmpeg, $outputTarget)) {
    Assert-PathsDisjoint -Left $inputPath -Right $canonicalRoot -Description 'canonical and caller'
}
Assert-PathsDisjoint -Left $outputTarget -Right $inputSource -Description 'output and source'
Assert-PathsDisjoint -Left $outputTarget -Right $inputCargoSeed -Description 'output and Cargo seed'
Assert-PathsDisjoint -Left $outputTarget -Right $inputFfmpeg -Description 'output and FFmpeg'

if (-not (Test-Path -LiteralPath (Join-Path $inputSource '.git'))) {
    throw "SourceRoot is not a Git checkout: $inputSource"
}
$gitOverrideEnvironmentNames = @(
    'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_OBJECT_DIRECTORY',
    'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_COMMON_DIR', 'GIT_NAMESPACE',
    'GIT_CEILING_DIRECTORIES', 'GIT_CONFIG_PARAMETERS', 'GIT_CONFIG_COUNT',
    'GIT_CONFIG_SYSTEM', 'GIT_CONFIG_GLOBAL', 'GIT_TEMPLATE_DIR', 'GIT_EXEC_PATH',
    'GIT_DEFAULT_HASH', 'GIT_DEFAULT_REF_FORMAT', 'GIT_ALLOW_PROTOCOL',
    'GIT_PROTOCOL_FROM_USER'
)
foreach ($environmentEntry in @(Get-ChildItem Env:)) {
    if (
        $gitOverrideEnvironmentNames -contains $environmentEntry.Name -or
        $environmentEntry.Name -match '^GIT_CONFIG_(KEY|VALUE)_[0-9]+$'
    ) {
        throw "ambient Git routing is not permitted: $($environmentEntry.Name)"
    }
}
$gitIsolationEnvironmentNames = @(
    'GIT_CONFIG_NOSYSTEM', 'GIT_CONFIG_GLOBAL', 'GIT_CONFIG_COUNT', 'GIT_ATTR_NOSYSTEM'
)
$initialGitEnvironment = @{}
foreach ($environmentName in $gitIsolationEnvironmentNames) {
    $initialGitEnvironment[$environmentName] = [Environment]::GetEnvironmentVariable($environmentName, 'Process')
}
$gitSafetyArguments = @('-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=NUL')
try {
    Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_NOSYSTEM' -Value '1'
    Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_GLOBAL' -Value 'NUL'
    Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_COUNT' -Value '0'
    Set-ProcessEnvironmentValue -Name 'GIT_ATTR_NOSYSTEM' -Value '1'
    $dirty = @(git @gitSafetyArguments -C $inputSource status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0 -or $dirty.Count -ne 0) {
        throw 'Reproducible builds require an entirely clean source checkout'
    }
    $actualSha = (git @gitSafetyArguments -C $inputSource rev-parse 'HEAD^{commit}').Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0 -or $actualSha -cne $GitSha.ToLowerInvariant()) {
        throw "Checkout SHA $actualSha does not match requested $GitSha"
    }
    $sourceTreeSha = (git @gitSafetyArguments -C $inputSource rev-parse 'HEAD^{tree}').Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0 -or $sourceTreeSha -notmatch '^[0-9a-f]{40}$') {
        throw 'source checkout did not report a canonical tree SHA'
    }
    $commitEpoch = (git @gitSafetyArguments -C $inputSource show -s --format=%ct $GitSha.ToLowerInvariant()).Trim()
    if ($LASTEXITCODE -ne 0 -or $commitEpoch -notmatch '^[0-9]+$' -or $commitEpoch -cne $SourceDateEpoch) {
        throw 'SOURCE_DATE_EPOCH must equal the exact source commit timestamp'
    }
    $unsafeTrackedEntries = @(git @gitSafetyArguments -C $inputSource ls-files -s | Select-String '^(120000|160000) ')
    if ($LASTEXITCODE -ne 0 -or $unsafeTrackedEntries.Count -ne 0) {
        throw 'tracked symlinks and gitlinks are not permitted in the canonical source archive'
    }
} finally {
    foreach ($environmentName in $gitIsolationEnvironmentNames) {
        Set-ProcessEnvironmentValue -Name $environmentName -Value $initialGitEnvironment[$environmentName]
    }
}

if (-not [string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable('CARGO_BUILD_TARGET', 'Process'))) {
    throw 'CARGO_BUILD_TARGET is not permitted for the reproducible Windows build'
}
$rustcVersion = (& rustup run 1.98.0 rustc -vV) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw 'rustup could not execute the reviewed rustc 1.98.0 toolchain'
}
$rustHostMatch = [regex]::Match($rustcVersion, '(?m)^host: ([A-Za-z0-9_.-]+)$')
if (-not $rustHostMatch.Success -or $rustHostMatch.Groups[1].Value -cne 'x86_64-pc-windows-msvc') {
    throw 'reproducible Windows releases require the x86_64-pc-windows-msvc host'
}
if (
    $rustcVersion -notmatch '(?m)^release: 1\.98\.0$' -or
    $rustcVersion -notmatch '(?m)^commit-hash: 88d9e12ae178fab0fb5cc050a94da85685d449ea$' -or
    $rustcVersion -notmatch '(?m)^LLVM version: 22\.1\.8$'
) {
    throw 'reproducible Windows releases require rustc 1.98.0'
}
$cargoVersion = (& rustup run 1.98.0 cargo -Vv) -join "`n"
if (
    $LASTEXITCODE -ne 0 -or
    $cargoVersion -notmatch '(?m)^release: 1\.98\.0$' -or
    $cargoVersion -notmatch '(?m)^commit-hash: 797e8a9bca276c1c9f9f738d2a20f484fa4eea9d$' -or
    $cargoVersion -notmatch '(?m)^host: x86_64-pc-windows-msvc$'
) {
    throw 'reproducible Windows releases require the reviewed Cargo 1.98.0 toolchain'
}
$nativeTarget = $rustHostMatch.Groups[1].Value
$nativeTargetUnderscored = $nativeTarget.Replace('-', '_').Replace('.', '_')

$libclangPath = [Environment]::GetEnvironmentVariable('LIBCLANG_PATH', 'Process')
if ([string]::IsNullOrWhiteSpace($libclangPath) -or -not (Test-Path -LiteralPath $libclangPath -PathType Container)) {
    throw 'LIBCLANG_PATH must name the reviewed LLVM directory'
}
$resolvedLibclangPath = (Resolve-Path -LiteralPath $libclangPath).Path
Assert-NoReparsePathChain -Path $resolvedLibclangPath
$llvmAr = Join-Path $resolvedLibclangPath 'llvm-ar.exe'
$libclang = Join-Path $resolvedLibclangPath 'libclang.dll'
if (-not (Test-Path -LiteralPath $llvmAr -PathType Leaf) -or (Split-Path -Leaf $llvmAr) -cne 'llvm-ar.exe') {
    throw 'reviewed lowercase llvm-ar.exe is missing'
}
if (-not (Test-Path -LiteralPath $libclang -PathType Leaf)) {
    throw 'reviewed libclang.dll is missing'
}
Assert-NoReparsePathChain -Path $llvmAr
Assert-NoReparsePathChain -Path $libclang
$llvmArHash = (Get-FileHash -LiteralPath $llvmAr -Algorithm SHA256).Hash.ToLowerInvariant()
if ($llvmArHash -cne '80934e8f208a0cc2a87a6057f871d0f492461952b8672464749a6c3dff34109c') {
    throw 'reviewed llvm-ar.exe SHA-256 mismatch'
}
$llvmArVersion = (& $llvmAr --version) -join "`n"
if ($LASTEXITCODE -ne 0 -or $llvmArVersion -notmatch '(?m)^  LLVM version 22\.1\.8$') {
    throw 'reviewed llvm-ar.exe version mismatch'
}
$libclangHash = (Get-FileHash -LiteralPath $libclang -Algorithm SHA256).Hash.ToLowerInvariant()
$libclangVersion = (Get-Item -LiteralPath $libclang -Force).VersionInfo.FileVersion
if (
    $libclangHash -cne '51fed10c43c3d31c1fe5bfe76bac60150970961e9b9b23cf014dbfcb5398bbfc' -or
    $libclangVersion -cne '22.1.8'
) {
    throw 'reviewed libclang.dll identity mismatch'
}

$controlledArchiverName = "AR_$nativeTarget"
$controlledAwsCmakeName = "AWS_LC_SYS_CMAKE_BUILDER_$nativeTargetUnderscored"
$controlledAwsSystemName = "AWS_LC_SYS_USE_SYSTEM_$nativeTargetUnderscored"
$archiverEnvironmentNames = @(
    "AR_$nativeTarget", "AR_$nativeTargetUnderscored", 'HOST_AR', 'TARGET_AR', 'AR',
    "ARFLAGS_$nativeTarget", "ARFLAGS_$nativeTargetUnderscored",
    'HOST_ARFLAGS', 'TARGET_ARFLAGS', 'ARFLAGS',
    "RANLIB_$nativeTarget", "RANLIB_$nativeTargetUnderscored", 'HOST_RANLIB', 'TARGET_RANLIB', 'RANLIB',
    "RANLIBFLAGS_$nativeTarget", "RANLIBFLAGS_$nativeTargetUnderscored",
    'HOST_RANLIBFLAGS', 'TARGET_RANLIBFLAGS', 'RANLIBFLAGS'
) | Select-Object -Unique
$nativeFlagEnvironmentNames = @(
    'HOST_CFLAGS', 'TARGET_CFLAGS', "CFLAGS_$nativeTarget", "CFLAGS_$nativeTargetUnderscored",
    "HOST_CFLAGS_$nativeTargetUnderscored", "TARGET_CFLAGS_$nativeTargetUnderscored",
    'HOST_CXXFLAGS', 'TARGET_CXXFLAGS', "CXXFLAGS_$nativeTarget", "CXXFLAGS_$nativeTargetUnderscored",
    "HOST_CXXFLAGS_$nativeTargetUnderscored", "TARGET_CXXFLAGS_$nativeTargetUnderscored",
    'AWS_LC_SYS_CFLAGS', "AWS_LC_SYS_CFLAGS_$nativeTargetUnderscored",
    'AWS_LC_SYS_HOST_CFLAGS', "AWS_LC_SYS_HOST_CFLAGS_$nativeTargetUnderscored",
    'AWS_LC_SYS_TARGET_CFLAGS', "AWS_LC_SYS_TARGET_CFLAGS_$nativeTargetUnderscored",
    'AWS_LC_SYS_CXXFLAGS', "AWS_LC_SYS_CXXFLAGS_$nativeTargetUnderscored",
    'AWS_LC_SYS_HOST_CXXFLAGS', "AWS_LC_SYS_HOST_CXXFLAGS_$nativeTargetUnderscored",
    'AWS_LC_SYS_TARGET_CXXFLAGS', "AWS_LC_SYS_TARGET_CXXFLAGS_$nativeTargetUnderscored",
    'LDFLAGS', 'HOST_LDFLAGS', 'TARGET_LDFLAGS',
    "LDFLAGS_$nativeTarget", "LDFLAGS_$nativeTargetUnderscored"
)
$compilerEnvironmentNames = @(
    'CC', 'CXX', "CC_$nativeTarget", "CC_$nativeTargetUnderscored",
    "CXX_$nativeTarget", "CXX_$nativeTargetUnderscored", 'HOST_CC', 'HOST_CXX',
    'RUSTC_LINKER', "CARGO_TARGET_$($nativeTargetUnderscored.ToUpperInvariant())_LINKER",
    'RUSTC', 'RUSTDOC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER',
    'CARGO_BUILD_RUSTC', 'CARGO_BUILD_RUSTC_WRAPPER', 'CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER',
    'RUSTC_BOOTSTRAP', 'RUSTDOCFLAGS', 'BINDGEN_EXTRA_CLANG_ARGS',
    "BINDGEN_EXTRA_CLANG_ARGS_$nativeTarget", "BINDGEN_EXTRA_CLANG_ARGS_$nativeTargetUnderscored",
    'CC_KNOWN_WRAPPER_CUSTOM', 'CRATE_CC_NO_DEFAULTS', 'CC_ENABLE_DEBUG_OUTPUT',
    'LINK', '_LINK_', 'RUSTFLAGS'
)
$cmakeEnvironmentNames = @(
    foreach ($cmakeVariable in @('CMAKE_GENERATOR', 'CMAKE_TOOLCHAIN_FILE')) {
        $cmakeVariable
        "${cmakeVariable}_$nativeTarget"
        "${cmakeVariable}_$nativeTargetUnderscored"
        "HOST_$cmakeVariable"
        "AWS_LC_SYS_$cmakeVariable"
        "AWS_LC_SYS_${cmakeVariable}_$nativeTargetUnderscored"
    }
) | Select-Object -Unique
$awsRouteVariables = @(
    'NO_PREFIX', 'PREGENERATING_BINDINGS', 'EXTERNAL_BINDGEN', 'NO_ASM', 'PREBUILT_NASM',
    'C_STD', 'CMAKE_BUILDER', 'NO_PREGENERATED_SRC', 'SMALL', 'EFFECTIVE_TARGET',
    'NO_JITTER_ENTROPY', 'NO_U1_BINDINGS', 'INCLUDES', 'SANITIZER', 'LINK_WHOLE_ARCHIVE',
    'STATIC', 'SYSTEM_DIR', 'USE_SYSTEM', 'SYSTEM_BINDINGS', 'SYSTEM_SKIP_VERSION_CHECK'
)
$awsRouteEnvironmentNames = @(
    foreach ($awsVariable in $awsRouteVariables) {
        "AWS_LC_SYS_$awsVariable"
        "AWS_LC_SYS_${awsVariable}_$nativeTargetUnderscored"
    }
) | Select-Object -Unique
$crateRouteEnvironmentNames = @(
    'DOCS_RS', 'RING_PREGENERATE_ASM', 'COLLIDE_BUILD_FFMPEG_BINARY',
    'COLLIDE_BUILD_FFPROBE_BINARY', 'PERL_EXECUTABLE', 'SPOUT2_SDK_DIR',
    'SPOUT2_LIB_DIR', 'CC_FORCE_DISABLE'
)
$rejectedEnvironmentNames = @(
    $archiverEnvironmentNames
    $nativeFlagEnvironmentNames
    $compilerEnvironmentNames
    $cmakeEnvironmentNames
    $awsRouteEnvironmentNames
    $crateRouteEnvironmentNames
) | Select-Object -Unique
foreach ($environmentName in $rejectedEnvironmentNames) {
    if ($null -ne [Environment]::GetEnvironmentVariable($environmentName, 'Process')) {
        throw "ambient build-tool configuration is not permitted: $environmentName"
    }
}
$allowedAmbientCargoNames = @(
    'CARGO_HOME', 'CARGO_TERM_COLOR', 'CARGO_INCREMENTAL', 'CARGO_ENCODED_RUSTFLAGS',
    'CARGO_TARGET_DIR', 'CARGO_NET_OFFLINE'
)
foreach ($environmentEntry in @(Get-ChildItem Env:)) {
    if (
        $environmentEntry.Name -like 'CARGO_*' -and
        $allowedAmbientCargoNames -notcontains $environmentEntry.Name
    ) {
        throw "unreviewed ambient Cargo configuration is not permitted: $($environmentEntry.Name)"
    }
    if ($environmentEntry.Name -like 'CMAKE_*') {
        throw "ambient CMake configuration is not permitted: $($environmentEntry.Name)"
    }
    if ($environmentEntry.Name -like 'AWS_LC_SYS_*') {
        throw "ambient AWS-LC routing is not permitted: $($environmentEntry.Name)"
    }
}
$saved = @{}
$environmentNames = @(
    'CARGO_HOME', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_TARGET_DIR', 'CARGO_NET_OFFLINE',
    'CARGO_INCREMENTAL', 'COLLIDE_BUILD_GIT_SHA', 'COLLIDE_BUILD_GIT_DIRTY',
    'COLLIDE_PUBLISHED_ARTIFACT', 'FFMPEG_DIR', 'FFMPEG_VERSION', 'SOURCE_DATE_EPOCH',
    'RUSTUP_TOOLCHAIN', 'GIT_CONFIG_NOSYSTEM', 'GIT_CONFIG_GLOBAL',
    'GIT_CONFIG_COUNT', 'GIT_ATTR_NOSYSTEM',
    'CC_SHELL_ESCAPED_FLAGS', 'CFLAGS', 'CXXFLAGS', 'CL', '_CL_', 'LINK', '_LINK_',
    'TEMP', 'TMP', 'TMPDIR', 'PATH', $controlledArchiverName,
    $controlledAwsCmakeName, $controlledAwsSystemName
) | Select-Object -Unique
foreach ($environmentName in $environmentNames) {
    $saved[$environmentName] = [Environment]::GetEnvironmentVariable($environmentName, 'Process')
}

$mutex = [Threading.Mutex]::new($false, $canonicalMutexName)
$mutexHeld = $false
$ownedStage = $false
$ownerStream = $null
$ownerNonce = [Guid]::NewGuid().ToString('N')
$stageCleanupSucceeded = $false
$environmentRestored = $false
$result = $null
$stagedExecutableBytes = $null

try {
    try {
        $mutexHeld = $mutex.WaitOne(0)
    } catch [Threading.AbandonedMutexException] {
        $mutexHeld = $true
    }
    if (-not $mutexHeld) {
        throw 'another canonical reproducible build owns the staging mutex'
    }
    if (Test-Path -LiteralPath $canonicalRoot) {
        throw 'canonical reproducible staging root already exists and will not be auto-deleted'
    }
    [CollideReproducibleNativePaths]::CreateNewDirectory($canonicalRoot)
    $ownedStage = $true
    $ownerStream = [IO.File]::Open($ownerPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $ownerBytes = [Text.Encoding]::UTF8.GetBytes("collide-repro-stage-v1`n$ownerNonce`n")
    $ownerStream.Write($ownerBytes, 0, $ownerBytes.Length)
    $ownerStream.Flush($true)
    foreach ($directory in @($canonicalSource, $canonicalCargoHome, $canonicalFfmpeg, $canonicalTemp)) {
        [CollideReproducibleNativePaths]::CreateNewDirectory($directory)
    }

    Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_NOSYSTEM' -Value '1'
    Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_GLOBAL' -Value 'NUL'
    Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_COUNT' -Value '0'
    Set-ProcessEnvironmentValue -Name 'GIT_ATTR_NOSYSTEM' -Value '1'
    git @gitSafetyArguments -C $inputSource archive --format=tar --output=$sourceArchive $GitSha.ToLowerInvariant()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $sourceArchive -PathType Leaf)) {
        throw 'failed to create the exact canonical source archive'
    }
    $sourceArchiveSha256 = (Get-FileHash -LiteralPath $sourceArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    & tar.exe -xf $sourceArchive -C $canonicalSource
    if ($LASTEXITCODE -ne 0) {
        throw 'failed to extract the canonical source archive'
    }
    Assert-NoReparsePoints -Path $canonicalSource -Recurse
    git @gitSafetyArguments -C $canonicalSource init --quiet --template=
    if ($LASTEXITCODE -ne 0) { throw 'failed to initialize canonical source Git metadata' }
    foreach ($gitConfiguration in @(
        @('core.autocrlf', 'false'), @('core.eol', 'lf'), @('core.symlinks', 'false'),
        @('core.fsmonitor', 'false'), @('core.untrackedCache', 'false'), @('core.hooksPath', 'NUL')
    )) {
        git @gitSafetyArguments -C $canonicalSource config $gitConfiguration[0] $gitConfiguration[1]
        if ($LASTEXITCODE -ne 0) { throw "failed to set canonical Git configuration $($gitConfiguration[0])" }
    }
    git @gitSafetyArguments -C $canonicalSource -c protocol.file.allow=always fetch --quiet --no-tags --no-write-fetch-head $inputSource $GitSha.ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw 'failed to import the exact source commit into canonical Git metadata' }
    git @gitSafetyArguments -C $canonicalSource symbolic-ref HEAD refs/heads/canonical
    if ($LASTEXITCODE -ne 0) { throw 'failed to bind canonical Git symbolic HEAD' }
    git @gitSafetyArguments -C $canonicalSource update-ref HEAD $GitSha.ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw 'failed to bind canonical Git commit' }
    git @gitSafetyArguments -C $canonicalSource read-tree $GitSha.ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw 'failed to bind canonical Git HEAD and index' }
    $canonicalSha = (git @gitSafetyArguments -C $canonicalSource rev-parse 'HEAD^{commit}').Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw 'failed to read canonical Git commit' }
    $canonicalTreeSha = (git @gitSafetyArguments -C $canonicalSource rev-parse 'HEAD^{tree}').Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw 'failed to read canonical Git tree' }
    $canonicalDirty = @(git @gitSafetyArguments -C $canonicalSource status --porcelain=v1 --untracked-files=all)
    if (
        $LASTEXITCODE -ne 0 -or
        $canonicalSha -cne $GitSha.ToLowerInvariant() -or
        $canonicalTreeSha -cne $sourceTreeSha -or
        $canonicalDirty.Count -ne 0
    ) {
        throw 'canonical source checkout disagrees with the exact clean input tree'
    }

    New-Item -ItemType Directory -Path (Join-Path $canonicalCargoHome 'registry') | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $canonicalCargoHome 'git') | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $canonicalCargoHome 'bin') | Out-Null
    $null = Copy-VerifiedTree -Source (Join-Path $inputCargoSeed 'registry\cache') -Destination (Join-Path $canonicalCargoHome 'registry\cache')
    $null = Copy-VerifiedTree -Source (Join-Path $inputCargoSeed 'registry\index') -Destination (Join-Path $canonicalCargoHome 'registry\index')
    $null = Copy-VerifiedTree -Source (Join-Path $inputCargoSeed 'git\db') -Destination (Join-Path $canonicalCargoHome 'git\db')
    $cargoAuditableSha256 = Copy-VerifiedFile -Source (Join-Path $inputCargoSeed 'bin\cargo-auditable.exe') -Destination (Join-Path $canonicalCargoHome 'bin\cargo-auditable.exe')
    $null = Copy-VerifiedFile -Source (Join-Path $inputCargoSeed '.crates.toml') -Destination (Join-Path $canonicalCargoHome '.crates.toml')
    $null = Copy-VerifiedFile -Source (Join-Path $inputCargoSeed '.crates2.json') -Destination (Join-Path $canonicalCargoHome '.crates2.json')
    $cargoSeedManifestSha256 = Get-TreeManifestDigest $canonicalCargoHome
    if (
        (Test-Path -LiteralPath (Join-Path $canonicalCargoHome 'registry\src')) -or
        (Test-Path -LiteralPath (Join-Path $canonicalCargoHome 'git\checkouts'))
    ) {
        throw 'expanded Cargo source/checkouts must be recreated offline, not copied from the caller'
    }

    foreach ($component in @('bin', 'include', 'lib')) {
        $null = Copy-VerifiedTree -Source (Join-Path $inputFfmpeg $component) -Destination (Join-Path $canonicalFfmpeg $component)
    }
    $ffmpegManifestSha256 = Get-TreeManifestDigest $canonicalFfmpeg
    Assert-NoCargoConfiguration -SourcePath $canonicalSource -CargoHomePath $canonicalCargoHome

    $source = [CollideReproducibleNativePaths]::GetLongPath($canonicalSource)
    $sourceShort = [CollideReproducibleNativePaths]::GetShortPath($source)
    $cargoHome = [CollideReproducibleNativePaths]::GetLongPath($canonicalCargoHome)
    $cargoHomeShort = [CollideReproducibleNativePaths]::GetShortPath($cargoHome)
    $ffmpeg = [CollideReproducibleNativePaths]::GetLongPath($canonicalFfmpeg)
    $ffmpegShort = [CollideReproducibleNativePaths]::GetShortPath($ffmpeg)
    $target = $canonicalTarget
    $targetShort = $canonicalTarget

    Set-ProcessEnvironmentValue -Name 'CARGO_HOME' -Value $cargoHome
    Set-ProcessEnvironmentValue -Name 'CARGO_TARGET_DIR' -Value $target
    Set-ProcessEnvironmentValue -Name 'CARGO_NET_OFFLINE' -Value 'true'
    Set-ProcessEnvironmentValue -Name 'CARGO_INCREMENTAL' -Value '0'
    Set-ProcessEnvironmentValue -Name 'RUSTUP_TOOLCHAIN' -Value '1.98.0'
    Set-ProcessEnvironmentValue -Name 'COLLIDE_BUILD_GIT_SHA' -Value $GitSha.ToLowerInvariant()
    Set-ProcessEnvironmentValue -Name 'COLLIDE_BUILD_GIT_DIRTY' -Value 'false'
    Set-ProcessEnvironmentValue -Name 'COLLIDE_PUBLISHED_ARTIFACT' -Value 'true'
    Set-ProcessEnvironmentValue -Name 'FFMPEG_DIR' -Value $ffmpeg
    Set-ProcessEnvironmentValue -Name 'FFMPEG_VERSION' -Value '9.0.1'
    Set-ProcessEnvironmentValue -Name 'SOURCE_DATE_EPOCH' -Value $SourceDateEpoch
    Set-ProcessEnvironmentValue -Name 'TEMP' -Value $canonicalTemp
    Set-ProcessEnvironmentValue -Name 'TMP' -Value $canonicalTemp
    Set-ProcessEnvironmentValue -Name 'TMPDIR' -Value $canonicalTemp
    Set-ProcessEnvironmentValue -Name $controlledArchiverName -Value $llvmAr
    Set-ProcessEnvironmentValue -Name $controlledAwsCmakeName -Value '0'
    Set-ProcessEnvironmentValue -Name $controlledAwsSystemName -Value '0'
    Set-ProcessEnvironmentValue -Name 'CL' -Value $null
    Set-ProcessEnvironmentValue -Name '_CL_' -Value $null
    Set-ProcessEnvironmentValue -Name 'LINK' -Value $null
    Set-ProcessEnvironmentValue -Name '_LINK_' -Value $null
    $buildPath = (Join-Path $cargoHome 'bin') + ';' + $saved['PATH']
    Set-ProcessEnvironmentValue -Name 'PATH' -Value $buildPath

    $installedCargoTools = (& rustup run 1.98.0 cargo install --list) -join "`n"
    if ($LASTEXITCODE -ne 0 -or $installedCargoTools -notmatch '(?m)^cargo-auditable v0\.7\.5:$') {
        throw 'cargo-auditable 0.7.5 is required in the canonical Cargo home'
    }
    Push-Location $source
    try {
        rustup run 1.98.0 cargo fetch --locked --offline
        if ($LASTEXITCODE -ne 0) { throw 'canonical offline cargo fetch failed' }
    } finally {
        Pop-Location
    }
    Assert-NoCargoConfiguration -SourcePath $source -CargoHomePath $cargoHome
    Assert-NoReparsePoints -Path $cargoHome -Recurse
    $cargoGitCheckoutReview = [ordered]@{
        'git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500' = '4b79500dfac64efcfb393eebc89f5c75565ee5ae'
        'git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500/crates/openfx-plugin/vendor/openfx' = '5aa788d5134f577c23eba158ded7592c4c471050'
    }
    $cargoCheckoutMetadataExclusions = @()
    foreach ($relativeCheckout in $cargoGitCheckoutReview.Keys) {
        $checkoutPath = Join-Path $cargoHome $relativeCheckout.Replace('/', '\')
        if (-not (Test-Path -LiteralPath $checkoutPath -PathType Container)) {
            throw "reviewed Cargo Git checkout is missing: $relativeCheckout"
        }
        $checkoutCommit = (git @gitSafetyArguments -C $checkoutPath rev-parse --verify 'HEAD^{commit}').Trim().ToLowerInvariant()
        if ($LASTEXITCODE -ne 0 -or $checkoutCommit -cne $cargoGitCheckoutReview[$relativeCheckout]) {
            throw "Cargo Git checkout commit differs from Cargo.lock: $relativeCheckout"
        }
        git @gitSafetyArguments -C $checkoutPath diff-index --quiet HEAD --
        if ($LASTEXITCODE -ne 0) {
            throw "Cargo Git checkout differs from its reviewed commit: $relativeCheckout"
        }
        $checkoutStatus = @(git @gitSafetyArguments -C $checkoutPath status --porcelain=v1 --untracked-files=all --ignore-submodules=dirty)
        if ($LASTEXITCODE -ne 0 -or $checkoutStatus.Count -ne 1 -or $checkoutStatus[0] -cne '?? .cargo-ok') {
            throw "Cargo Git checkout has unexpected untracked state: $relativeCheckout"
        }
        $cargoOk = Join-Path $checkoutPath '.cargo-ok'
        if (-not (Test-Path -LiteralPath $cargoOk -PathType Leaf) -or (Get-Item -LiteralPath $cargoOk -Force).Length -ne 0) {
            throw "Cargo Git checkout marker is not the expected empty file: $relativeCheckout"
        }
        Assert-OnlyDefaultDataStream -Path $cargoOk
        $relativeMetadataFiles = @(
            "$relativeCheckout/.git/index",
            "$relativeCheckout/.git/logs/HEAD",
            "$relativeCheckout/.git/logs/refs/heads/master"
        )
        foreach ($relativeMetadata in $relativeMetadataFiles) {
            $metadataPath = Join-Path $cargoHome $relativeMetadata.Replace('/', '\')
            if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
                throw "reviewed Cargo Git bookkeeping file is missing: $relativeMetadata"
            }
            Assert-OnlyDefaultDataStream -Path $metadataPath
            $cargoCheckoutMetadataExclusions += $relativeMetadata
        }
    }
    $observedCargoCheckoutMetadata = @(
        Get-ChildItem -LiteralPath (Join-Path $cargoHome 'git\checkouts') -Force -Recurse -File |
            ForEach-Object { $_.FullName.Substring($cargoHome.Length).TrimStart([char[]]@('\', '/')).Replace('\', '/') } |
            Where-Object { $_ -match '/\.git/index$' -or $_ -match '/\.git/logs/' } |
            Sort-Object
    )
    if (@(Compare-Object @($cargoCheckoutMetadataExclusions) $observedCargoCheckoutMetadata -SyncWindow 0).Count -ne 0) {
        throw 'Cargo Git checkout metadata inventory differs from the reviewed nondeterministic bookkeeping set'
    }
    $cargoBookkeepingExclusions = @(
        '.global-cache', '.package-cache', '.package-cache-mutate'
    ) + @($cargoCheckoutMetadataExclusions)
    $unexpectedCargoRootFiles = @(
        Get-ChildItem -LiteralPath $cargoHome -Force -File |
            Where-Object Name -notin @('.crates.toml', '.crates2.json', '.global-cache', '.package-cache', '.package-cache-mutate')
    )
    if ($unexpectedCargoRootFiles.Count -ne 0) {
        throw 'canonical Cargo home contains an unreviewed root-level file'
    }
    $cargoExpandedManifestSha256 = Get-TreeManifestDigest -Path $cargoHome -ExcludedRelativePaths $cargoBookkeepingExclusions

    $unitSeparator = [char]0x1f
    $remappedSource = $source.Replace('\', '/')
    $remappedTarget = $target.Replace('\', '/')
    $remappedCargoHome = $cargoHome.Replace('\', '/')
    $remappedFfmpeg = $ffmpeg.Replace('\', '/')
    $remappedFfmpegShort = $ffmpegShort.Replace('\', '/')
    $encodedFlags = @(
        '-C', 'link-arg=/Brepro',
        '-C', 'link-arg=/DEBUG:NONE',
        "--remap-path-prefix=$remappedSource=/collide-o-scope",
        "--remap-path-prefix=$remappedTarget=/collide-o-scope-target",
        "--remap-path-prefix=$remappedCargoHome=/cargo-home",
        "--remap-path-prefix=$remappedFfmpeg=/ffmpeg",
        "--remap-path-prefix=$remappedFfmpegShort=/ffmpeg"
    ) -join $unitSeparator
    Set-ProcessEnvironmentValue -Name 'CARGO_ENCODED_RUSTFLAGS' -Value $encodedFlags

    $nativeTrimArguments = @(
        "/d1trimfile:$source",
        "/d1trimfile:$sourceShort",
        "/d1trimfile:$target",
        "/d1trimfile:$targetShort",
        "/d1trimfile:$cargoHome",
        "/d1trimfile:$cargoHomeShort",
        "/d1trimfile:$ffmpeg",
        "/d1trimfile:$ffmpegShort"
    ) | Select-Object -Unique
    $nativeTrimFlags = ($nativeTrimArguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
    Set-ProcessEnvironmentValue -Name 'CC_SHELL_ESCAPED_FLAGS' -Value '1'
    Set-ProcessEnvironmentValue -Name 'CFLAGS' -Value $nativeTrimFlags
    Set-ProcessEnvironmentValue -Name 'CXXFLAGS' -Value $nativeTrimFlags

    Push-Location $source
    try {
        rustup run 1.98.0 cargo auditable build --locked --offline --release --bin collide-o-scope
        if ($LASTEXITCODE -ne 0) { throw 'cargo auditable build failed' }
    } finally {
        Pop-Location
    }

    $executable = Join-Path $target 'release\collide-o-scope.exe'
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "expected canonical release executable is missing: $executable"
    }
    if (@(Get-ChildItem -LiteralPath (Join-Path $target 'release') -File -Filter '*.pdb').Count -ne 0) {
        throw 'canonical release directory contains a PDB despite /DEBUG:NONE'
    }
    Assert-PortableExecutableHasNoCodeView -Executable $executable
    $executableBytes = [IO.File]::ReadAllBytes($executable)
    $latin1 = [Text.Encoding]::GetEncoding(28591)
    $binaryView = $latin1.GetString($executableBytes)
    $needleEncodings = @([Text.Encoding]::UTF8, [Text.Encoding]::Unicode, [Text.Encoding]::BigEndianUnicode)
    $profilesRoot = (Split-Path -Parent $userProfile).TrimEnd([char[]]@('\', '/')) + '\'
    $builderSpecificPaths = @(
        $inputSource, $inputSourceShort, $inputCargoSeed, $inputCargoSeedShort,
        $inputFfmpeg, $inputFfmpegShort, $outputTarget,
        $source, $sourceShort, $target, $targetShort, $cargoHome, $cargoHomeShort,
        $ffmpeg, $ffmpegShort, $canonicalRoot, $canonicalTemp,
        $userProfile, $userProfileShort, $profilesRoot, 'C:\Users\'
    ) | Select-Object -Unique
    foreach ($builderSpecificPath in $builderSpecificPaths) {
        foreach ($spelling in @($builderSpecificPath, $builderSpecificPath.Replace('\', '/')) | Select-Object -Unique) {
            foreach ($encoding in $needleEncodings) {
                $needleView = $latin1.GetString($encoding.GetBytes($spelling))
                if ($binaryView.IndexOf($needleView, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                    throw 'release executable contains a builder-specific path'
                }
            }
        }
    }
    try {
        Set-ProcessEnvironmentValue -Name 'PATH' -Value ((Join-Path $ffmpeg 'bin') + ';' + $buildPath)
        $identityJson = & $executable --version-json
        if ($LASTEXITCODE -ne 0) { throw 'built executable rejected --version-json' }
    } finally {
        Set-ProcessEnvironmentValue -Name 'PATH' -Value $buildPath
    }
    $identity = $identityJson | ConvertFrom-Json
    $packageVersionMatch = [regex]::Match(
        [IO.File]::ReadAllText((Join-Path $source 'Cargo.toml'), [Text.Encoding]::UTF8),
        '(?m)^version = "([0-9]+\.[0-9]+\.[0-9]+)"$'
    )
    if (-not $packageVersionMatch.Success) { throw 'canonical Cargo package version is missing' }
    $cargoLockHash = (Get-FileHash -LiteralPath (Join-Path $source 'Cargo.lock') -Algorithm SHA256).Hash.ToLowerInvariant()
    $ffmpegBinaryHash = (Get-FileHash -LiteralPath (Join-Path $ffmpeg 'bin\ffmpeg.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
    $ffprobeBinaryHash = (Get-FileHash -LiteralPath (Join-Path $ffmpeg 'bin\ffprobe.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedFfmpegLibraries = 'avcodec-63.dll,avdevice-63.dll,avfilter-12.dll,avformat-63.dll,avutil-61.dll,ffmpeg=9.0.1,swresample-7.dll,swscale-10.dll'
    $identityRustcVv = [regex]::Replace($rustcVersion, '(?m)^rustc 1\.98\.0 .+$', 'rustc 1.98.0')
    if (
        $identity.schema_version -ne 1 -or
        $identity.package_name -cne 'collide-o-scope' -or
        $identity.version -cne $packageVersionMatch.Groups[1].Value -or
        $identity.git_sha -cne $GitSha.ToLowerInvariant() -or
        $identity.git_dirty -or $identity.profile -cne 'release' -or
        $identity.target -cne $nativeTarget -or $identity.enabled_features -cne '(none)' -or
        $identity.rustc_vv -cne $identityRustcVv -or $identity.cargo_version -cne 'cargo 1.98.0' -or
        $identity.ffmpeg_libraries -cne $expectedFfmpegLibraries -or
        $identity.ffmpeg_binary_version -cne 'ffmpeg version 9.0.1' -or
        $identity.ffmpeg_binary_sha256 -cne $ffmpegBinaryHash -or
        $identity.ffprobe_binary_version -cne 'ffprobe version 9.0.1' -or
        $identity.ffprobe_binary_sha256 -cne $ffprobeBinaryHash -or
        $identity.cargo_lock_sha256 -cne $cargoLockHash -or
        -not $identity.published_artifact
    ) {
        throw 'release executable embeds an incomplete or unexpected BuildIdentity'
    }
    $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($executableBytes)).ToLowerInvariant()
    if ((Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant() -cne $hash) {
        throw 'canonical executable changed after its verified bytes were buffered'
    }
    $stagedExecutableBytes = $executableBytes

    if (
        (Get-FileHash -LiteralPath $llvmAr -Algorithm SHA256).Hash.ToLowerInvariant() -cne $llvmArHash -or
        (Get-FileHash -LiteralPath $libclang -Algorithm SHA256).Hash.ToLowerInvariant() -cne $libclangHash
    ) {
        throw 'reviewed LLVM inputs changed during the build'
    }

    $canonicalShaAfter = (git @gitSafetyArguments -C $source rev-parse 'HEAD^{commit}').Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw 'failed to re-read canonical Git commit after the build' }
    $canonicalTreeAfter = (git @gitSafetyArguments -C $source rev-parse 'HEAD^{tree}').Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw 'failed to re-read canonical Git tree after the build' }
    $canonicalDirtyAfter = @(git @gitSafetyArguments -C $source status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0) { throw 'failed to re-check canonical Git cleanliness after the build' }
    $inputShaAfter = (git @gitSafetyArguments -C $inputSource rev-parse 'HEAD^{commit}').Trim().ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw 'failed to re-read input Git commit after the build' }
    $inputDirtyAfter = @(git @gitSafetyArguments -C $inputSource status --porcelain=v1 --untracked-files=all)
    if (
        $LASTEXITCODE -ne 0 -or $canonicalShaAfter -cne $GitSha.ToLowerInvariant() -or
        $canonicalTreeAfter -cne $sourceTreeSha -or $canonicalDirtyAfter.Count -ne 0 -or
        $inputShaAfter -cne $GitSha.ToLowerInvariant() -or $inputDirtyAfter.Count -ne 0
    ) {
        throw 'source identity changed during the canonical build'
    }
    Assert-NoReparsePoints -Path $canonicalRoot -Recurse
    $ownerStream.Position = 0
    $ownerCheckBytes = [byte[]]::new($ownerBytes.Length)
    if ($ownerStream.Read($ownerCheckBytes, 0, $ownerCheckBytes.Length) -ne $ownerBytes.Length) {
        throw 'canonical stage owner sentinel was truncated'
    }
    if (-not [Security.Cryptography.CryptographicOperations]::FixedTimeEquals($ownerBytes, $ownerCheckBytes)) {
        throw 'canonical stage owner sentinel changed during the build'
    }
    $result = [ordered]@{
        schema_version = 1
        contract = 'collide-windows-canonical-repro-v1'
        wrapper_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        source_commit = $GitSha.ToLowerInvariant()
        source_tree_sha = $sourceTreeSha
        source_date_epoch = $SourceDateEpoch
        source_archive_sha256 = $sourceArchiveSha256
        cargo_seed_manifest_sha256 = $cargoSeedManifestSha256
        cargo_expanded_manifest_sha256 = $cargoExpandedManifestSha256
        cargo_manifest_exclusions = @($cargoBookkeepingExclusions)
        cargo_auditable_sha256 = $cargoAuditableSha256
        ffmpeg_manifest_sha256 = $ffmpegManifestSha256
        ffmpeg_version = '9.0.1'
        rustc_version = '1.98.0'
        rustc_commit = '88d9e12ae178fab0fb5cc050a94da85685d449ea'
        cargo_version = '1.98.0'
        cargo_commit = '797e8a9bca276c1c9f9f738d2a20f484fa4eea9d'
        llvm_ar_sha256 = $llvmArHash
        llvm_ar_version = '22.1.8'
        libclang_sha256 = $libclangHash
        libclang_version = $libclangVersion
        offline = $true
        canonical_root = $canonicalRoot
        executable_sha256 = $hash
        identity_sha256 = $identity.identity_sha256
        cleanup_succeeded = $false
    }
} finally {
    $restoreFailure = $null
    foreach ($environmentName in $environmentNames) {
        try {
            Set-ProcessEnvironmentValue -Name $environmentName -Value $saved[$environmentName]
        } catch {
            if ($null -eq $restoreFailure) { $restoreFailure = $_ }
        }
    }
    $environmentRestored = $null -eq $restoreFailure
    $cleanupFailure = $null
    if ($null -ne $ownerStream) {
        try { $ownerStream.Dispose() } catch { $cleanupFailure = $_ }
        $ownerStream = $null
    }
    if ($ownedStage) {
        try {
            $resolvedCanonicalRoot = (Resolve-Path -LiteralPath $canonicalRoot).Path.TrimEnd([char[]]@('\', '/'))
            if ($resolvedCanonicalRoot -cne $canonicalRoot) {
                throw "canonical stage resolved unexpectedly: $resolvedCanonicalRoot"
            }
            Assert-NoReparsePoints -Path $canonicalRoot -Recurse
            $ownerText = [IO.File]::ReadAllText($ownerPath, [Text.Encoding]::UTF8)
            if ($ownerText -cne "collide-repro-stage-v1`n$ownerNonce`n") {
                throw 'canonical stage ownership sentinel mismatch during cleanup'
            }
            Remove-Item -LiteralPath $canonicalRoot -Recurse -Force
            if (Test-Path -LiteralPath $canonicalRoot) {
                throw 'canonical stage cleanup left the root present'
            }
            $stageCleanupSucceeded = $true
        } catch {
            if ($null -eq $cleanupFailure) { $cleanupFailure = $_ }
        }
    }
    if ($mutexHeld) {
        try { $mutex.ReleaseMutex() } catch { if ($null -eq $cleanupFailure) { $cleanupFailure = $_ } }
    }
    $mutex.Dispose()
    if ($null -ne $restoreFailure) { throw $restoreFailure }
    if ($null -ne $cleanupFailure) { throw $cleanupFailure }
}

if ($null -eq $result -or $null -eq $stagedExecutableBytes -or -not $environmentRestored -or -not $stageCleanupSucceeded) {
    throw 'canonical build did not complete its cleanup and restoration transaction'
}
if (Test-Path -LiteralPath $outputTarget) {
    throw 'TargetDir appeared before cleanup-complete publication'
}
$publishNonce = [Guid]::NewGuid().ToString('N')
$publishRoot = Join-Path $outputParent ".collide-publish-$publishNonce"
$publishMoved = $false
try {
    [CollideReproducibleNativePaths]::CreateNewDirectory($publishRoot)
    $publishRelease = Join-Path $publishRoot 'release'
    [CollideReproducibleNativePaths]::CreateNewDirectory($publishRelease)
    $publishExecutable = Join-Path $publishRelease 'collide-o-scope.exe'
    $publishStream = [IO.File]::Open($publishExecutable, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $publishStream.Write($stagedExecutableBytes, 0, $stagedExecutableBytes.Length)
        $publishStream.Flush($true)
    } finally {
        $publishStream.Dispose()
    }
    $publishedHash = (Get-FileHash -LiteralPath $publishExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($publishedHash -cne $result.executable_sha256) {
        throw 'publication staging bytes differ from the verified canonical executable'
    }
    [IO.Directory]::Move($publishRoot, $outputTarget)
    $publishMoved = $true
} finally {
    if (-not $publishMoved -and (Test-Path -LiteralPath $publishRoot)) {
        $resolvedPublishRoot = (Resolve-Path -LiteralPath $publishRoot).Path
        $publishBoundary = (Get-NormalizedFullPath $outputParent) + '\.collide-publish-'
        if (-not $resolvedPublishRoot.StartsWith($publishBoundary, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'refusing cleanup of an unexpected publication staging path'
        }
        Assert-NoReparsePoints -Path $publishRoot -Recurse
        Remove-Item -LiteralPath $publishRoot -Recurse -Force
    }
}

$finalExecutable = Join-Path $outputTarget 'release\collide-o-scope.exe'
if (-not (Test-Path -LiteralPath $finalExecutable -PathType Leaf)) {
    throw 'cleanup-complete publication did not create the final executable'
}
$finalHash = (Get-FileHash -LiteralPath $finalExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
if ($finalHash -cne $result.executable_sha256) {
    throw 'published executable differs from the verified canonical bytes'
}
$result.cleanup_succeeded = $true
$result.executable = $finalExecutable
$result | ConvertTo-Json -Compress
