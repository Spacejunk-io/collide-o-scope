# Windows build helper for collide-o-scope.
#
# Prerequisites (one-time, all via winget):
#   winget install -e --id Gyan.FFmpeg.Shared --version 9.0.1   # exact FFmpeg 9 shared SDK
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

# --- Locate the exact FFmpeg shared SDK ---
$expectedFfmpegVersion = "9.0.1"
$expectedFfmpegDirectory = "ffmpeg-$expectedFfmpegVersion-full_build-shared"
if (-not [string]::IsNullOrWhiteSpace($env:FFMPEG_DIR)) {
    $ffmpegDir = Get-Item -LiteralPath ([IO.Path]::GetFullPath($env:FFMPEG_DIR)) -ErrorAction Stop
    if (-not $ffmpegDir.PSIsContainer) {
        throw "FFMPEG_DIR must identify an FFmpeg SDK directory: $($ffmpegDir.FullName)"
    }
} else {
    $wingetPackages = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    $ffmpegDirs = @(
        Get-ChildItem -LiteralPath $wingetPackages -Filter "Gyan.FFmpeg.Shared_*" -Directory -ErrorAction SilentlyContinue |
            ForEach-Object {
                Get-ChildItem -LiteralPath $_.FullName -Filter $expectedFfmpegDirectory -Directory -Recurse -ErrorAction SilentlyContinue
            } |
            Sort-Object FullName -Unique
    )
    if ($ffmpegDirs.Count -eq 0) {
        throw "The exact FFmpeg $expectedFfmpegVersion shared SDK was not found. Run: winget install -e --id Gyan.FFmpeg.Shared --version $expectedFfmpegVersion"
    }
    if ($ffmpegDirs.Count -ne 1) {
        $locations = ($ffmpegDirs.FullName | ForEach-Object { "  $_" }) -join "`n"
        throw "Multiple exact FFmpeg $expectedFfmpegVersion SDKs were found. Set FFMPEG_DIR to one reviewed directory:`n$locations"
    }
    $ffmpegDir = $ffmpegDirs[0]
}

$requiredFfmpegPaths = @(
    "include\libavcodec\avcodec.h",
    "lib\avcodec.lib",
    "bin\ffmpeg.exe",
    "bin\ffprobe.exe"
)
foreach ($relativePath in $requiredFfmpegPaths) {
    if (-not (Test-Path -LiteralPath (Join-Path $ffmpegDir.FullName $relativePath) -PathType Leaf)) {
        throw "FFMPEG_DIR is not a complete shared SDK; missing $relativePath under $($ffmpegDir.FullName)"
    }
}
$expectedFfmpegDlls = @(
    "avcodec-63.dll",
    "avdevice-63.dll",
    "avfilter-12.dll",
    "avformat-63.dll",
    "avutil-61.dll",
    "swresample-7.dll",
    "swscale-10.dll"
)
$observedFfmpegDlls = @(
    Get-ChildItem -LiteralPath (Join-Path $ffmpegDir.FullName "bin") -File |
        Where-Object { $_.Extension -ieq ".dll" } |
        ForEach-Object Name |
        Sort-Object
)
if (@(Compare-Object $expectedFfmpegDlls $observedFfmpegDlls -SyncWindow 0).Count -ne 0) {
    throw "FFMPEG_DIR must contain exactly the seven FFmpeg 9.0.1 ABI DLLs"
}
$ffmpegExecutable = Join-Path $ffmpegDir.FullName "bin\ffmpeg.exe"
# Preserve the native exit code before any pipeline runs. Windows PowerShell 5.1
# can replace it when Select-Object consumes the native command directly.
$ffmpegVersionOutput = @(& $ffmpegExecutable -hide_banner -version 2>&1)
$ffmpegExitCode = $LASTEXITCODE
$ffmpegVersionLine = ($ffmpegVersionOutput | Select-Object -First 1).ToString().Trim()
if ($ffmpegExitCode -ne 0 -or $ffmpegVersionLine -notmatch '^ffmpeg version 9\.0\.1(?:[- ].*)?$') {
    throw "FFMPEG_DIR must contain FFmpeg 9.0.1; observed '$ffmpegVersionLine'"
}
$ffprobeExecutable = Join-Path $ffmpegDir.FullName "bin\ffprobe.exe"
$ffprobeVersionOutput = @(& $ffprobeExecutable -hide_banner -version 2>&1)
$ffprobeExitCode = $LASTEXITCODE
$ffprobeVersionLine = ($ffprobeVersionOutput | Select-Object -First 1).ToString().Trim()
if ($ffprobeExitCode -ne 0 -or $ffprobeVersionLine -notmatch '^ffprobe version 9\.0\.1(?:[- ].*)?$') {
    throw "FFMPEG_DIR must contain ffprobe 9.0.1; observed '$ffprobeVersionLine'"
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
if ($Run) {
    # Cargo cannot replace an executable that is still running on Windows.
    # Restart only this checkout/profile's executable; never stop another
    # collide-o-scope build that may be running from a different folder.
    function Test-ExactExecutableProcess {
        param(
            [System.Diagnostics.Process]$Candidate,
            [string]$ExpectedPath
        )
        try {
            if ($Candidate.HasExited) { return $false }
            $candidatePath = [System.IO.Path]::GetFullPath($Candidate.Path)
            return [System.StringComparer]::OrdinalIgnoreCase.Equals($candidatePath, $ExpectedPath)
        } catch {
            # A process that exited while being inspected is no longer a lock.
            return $false
        }
    }

    $targetRoot = if ([string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
        Join-Path $projectRoot "target"
    } elseif ([System.IO.Path]::IsPathRooted($env:CARGO_TARGET_DIR)) {
        $env:CARGO_TARGET_DIR
    } else {
        Join-Path $projectRoot $env:CARGO_TARGET_DIR
    }
    $profileName = if ($Release) { "release" } else { "debug" }
    $executablePath = [System.IO.Path]::GetFullPath((Join-Path $targetRoot "$profileName\collide-o-scope.exe"))
    $runningCopies = @(Get-Process -Name "collide-o-scope" -ErrorAction SilentlyContinue | Where-Object {
        try { [System.IO.Path]::GetFullPath($_.Path) -eq $executablePath } catch { $false }
    })
    foreach ($process in $runningCopies) {
        if (-not (Test-ExactExecutableProcess $process $executablePath)) { continue }
        Write-Host "Stopping prior run PID $($process.Id) so the executable can be rebuilt..."
        try {
            $closeRequested = $process.CloseMainWindow()
        } catch {
            if (Test-ExactExecutableProcess $process $executablePath) {
                throw "Unable to close the prior collide-o-scope run at '$executablePath' (PID $($process.Id)). Close it manually and retry."
            }
            continue
        }
        if (-not $closeRequested) {
            try {
                # Kill through the already-validated Process object. Do not
                # look the PID up again: Windows may reuse an exited PID.
                $process.Kill()
            } catch {
                if (Test-ExactExecutableProcess $process $executablePath) {
                    throw "Unable to stop the prior collide-o-scope run at '$executablePath' (PID $($process.Id)). Close it manually and retry."
                }
            }
        }
    }
    foreach ($process in $runningCopies) {
        # Windows PowerShell 5.1 lacks a bounded process-wait command. Poll the
        # exact validated Process object briefly, never a newly reused PID.
        $deadline = [DateTime]::UtcNow.AddSeconds(5)
        while ((Test-ExactExecutableProcess $process $executablePath) -and [DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Milliseconds 100
        }
        if (Test-ExactExecutableProcess $process $executablePath) {
            try {
                $process.Kill()
            } catch {
                throw "Unable to force-stop the prior collide-o-scope run at '$executablePath' (PID $($process.Id)). Close it manually and retry."
            }
            # GPU/worker teardown can release the image lock a moment after
            # the process accepts termination. Keep this bounded, but allow
            # enough time for Windows to retire the executable mapping.
            $forceDeadline = [DateTime]::UtcNow.AddSeconds(10)
            while ((Test-ExactExecutableProcess $process $executablePath) -and [DateTime]::UtcNow -lt $forceDeadline) {
                Start-Sleep -Milliseconds 100
            }
            if (Test-ExactExecutableProcess $process $executablePath) {
                throw "The prior collide-o-scope run at '$executablePath' (PID $($process.Id)) did not exit. Close it manually and retry."
            }
        }
    }

    # Fail closed if a new copy of this same executable appeared while the old
    # process was stopping. Never terminate a process that was not validated in
    # the original enumeration.
    $remainingExactCopies = @(Get-Process -Name "collide-o-scope" -ErrorAction SilentlyContinue | Where-Object {
        try {
            [System.StringComparer]::OrdinalIgnoreCase.Equals(
                [System.IO.Path]::GetFullPath($_.Path),
                $executablePath
            )
        } catch { $false }
    })
    if ($remainingExactCopies.Count -gt 0) {
        $remainingPids = ($remainingExactCopies | ForEach-Object { $_.Id }) -join ", "
        throw "A collide-o-scope process still holds '$executablePath' (PID $remainingPids). Close it manually and retry."
    }
    $cargoCmd = "cargo run $profileFlag"
}

Write-Host "FFMPEG_DIR    = $($ffmpegDir.FullName)"
Write-Host "LIBCLANG_PATH = $libclang"
Write-Host "Building via  : $vcvars"

$env:FFMPEG_DIR = $ffmpegDir.FullName
$env:FFMPEG_VERSION = $expectedFfmpegVersion
$env:LIBCLANG_PATH = $libclang
$env:PATH = (Join-Path $ffmpegDir.FullName "bin") + ";" + $env:PATH
cmd /d /s /c "`"$vcvars`" >nul && $cargoCmd"
exit $LASTEXITCODE
