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

cmd /c "`"$vcvars`" >nul && set FFMPEG_DIR=$($ffmpegDir.FullName)&& set LIBCLANG_PATH=$libclang&& $cargoCmd"
exit $LASTEXITCODE
