# Windows build helper for collide-o-scope.
#
# Prerequisites (one-time, all via winget):
#   winget install -e --id Gyan.FFmpeg.Shared --version 8.1.2   # ffmpeg 8.x shared dev libs (must match ffmpeg-next major)
#   winget install -e --id LLVM.LLVM                            # libclang for bindgen
#   Visual Studio 2022 with the "Desktop development with C++" workload
#
# The MSVC environment (vcvars64) is only needed when ffmpeg-sys-next
# regenerates its bindings (first build or after `cargo clean`); afterwards
# a plain `cargo build` works too.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1 [-Release] [-Run]

param(
    [switch]$Release,
    [switch]$Run
)

$ErrorActionPreference = "Stop"

# --- Run from the project root, wherever the script was invoked from ---
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
Write-Host "Project root  = $projectRoot"

# --- Locate ffmpeg shared dev libs (winget install location) ---
$ffmpegPkg = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "Gyan.FFmpeg.Shared_*" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $ffmpegPkg) {
    Write-Error "ffmpeg shared package not found. Run: winget install -e --id Gyan.FFmpeg.Shared --version 8.1.2"
}
$ffmpegDir = Get-ChildItem $ffmpegPkg.FullName -Filter "ffmpeg-*-shared" -Directory | Select-Object -First 1
if ($null -eq $ffmpegDir) {
    Write-Error "ffmpeg build directory not found inside $($ffmpegPkg.FullName)"
}
if ($ffmpegDir.Name -notmatch "^ffmpeg-8\.") {
    Write-Warning "Found $($ffmpegDir.Name) but Cargo.toml pins ffmpeg-next = `"8`" (needs ffmpeg 8.x). Install with: winget install -e --id Gyan.FFmpeg.Shared --version 8.1.2"
}

# --- Locate libclang ---
$libclang = "C:\Program Files\LLVM\bin"
if (-not (Test-Path "$libclang\libclang.dll")) {
    Write-Error "libclang.dll not found. Run: winget install -e --id LLVM.LLVM"
}

# --- Locate vcvars64 ---
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ([string]::IsNullOrEmpty($vsPath)) {
    Write-Error "Visual Studio C++ tools not found. Install the 'Desktop development with C++' workload."
}
$vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"

$profileFlag = ""
if ($Release) { $profileFlag = "--release" }
$cargoCmd = "cargo build $profileFlag"
if ($Run) { $cargoCmd = "cargo run $profileFlag" }

Write-Host "FFMPEG_DIR    = $($ffmpegDir.FullName)"
Write-Host "LIBCLANG_PATH = $libclang"
Write-Host "Building via  : $vcvars"

cmd /c "`"$vcvars`" >nul && set FFMPEG_DIR=$($ffmpegDir.FullName)&& set LIBCLANG_PATH=$libclang&& $cargoCmd"
exit $LASTEXITCODE
