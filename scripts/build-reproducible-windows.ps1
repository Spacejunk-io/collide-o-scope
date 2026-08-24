param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$TargetDir,
    [Parameter(Mandatory = $true)][string]$FfmpegDir,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{40}$')][string]$GitSha,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9]+$')][string]$SourceDateEpoch
)

$ErrorActionPreference = "Stop"

$source = (Resolve-Path -LiteralPath $SourceRoot).Path
$ffmpeg = (Resolve-Path -LiteralPath $FfmpegDir).Path
$target = [System.IO.Path]::GetFullPath($TargetDir)
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

$installedCargoTools = (cargo install --list) -join "`n"
if (
    $LASTEXITCODE -ne 0 -or
    $installedCargoTools -notmatch '(?m)^cargo-auditable v0\.7\.5:$'
) {
    throw "cargo-auditable 0.7.5 is required"
}

$saved = @{}
$names = @(
    "CARGO_ENCODED_RUSTFLAGS", "CARGO_TARGET_DIR", "COLLIDE_BUILD_GIT_SHA",
    "COLLIDE_BUILD_GIT_DIRTY", "COLLIDE_PUBLISHED_ARTIFACT", "FFMPEG_DIR",
    "SOURCE_DATE_EPOCH", "PATH"
)
foreach ($name in $names) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    $unitSeparator = [char]0x1f
    $remappedSource = $source.Replace('\', '/')
    $remappedTarget = $target.Replace('\', '/')
    $encodedFlags = @(
        "-C", "link-arg=/Brepro",
        "--remap-path-prefix=$remappedSource=/collide-o-scope",
        "--remap-path-prefix=$remappedTarget=/collide-o-scope-target"
    ) -join $unitSeparator
    $env:CARGO_ENCODED_RUSTFLAGS = $encodedFlags
    $env:CARGO_TARGET_DIR = $target
    $env:COLLIDE_BUILD_GIT_SHA = $GitSha.ToLowerInvariant()
    $env:COLLIDE_BUILD_GIT_DIRTY = "false"
    $env:COLLIDE_PUBLISHED_ARTIFACT = "true"
    $env:FFMPEG_DIR = $ffmpeg
    $env:SOURCE_DATE_EPOCH = $SourceDateEpoch
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
