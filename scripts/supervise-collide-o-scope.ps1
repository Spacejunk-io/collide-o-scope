[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ApplicationArguments,
    [ValidateRange(0, 1)]
    [int]$MaximumGpuRestarts = 1
)

$ErrorActionPreference = 'Stop'
$resolvedExecutable = (Resolve-Path -LiteralPath $Executable).Path
if (-not (Test-Path -LiteralPath $resolvedExecutable -PathType Leaf)) {
    throw "collide-o-scope executable is not a file: $resolvedExecutable"
}

$gpuRestarts = 0
while ($true) {
    & $resolvedExecutable @ApplicationArguments
    $applicationExitCode = $LASTEXITCODE
    if ($applicationExitCode -ne 75) {
        exit $applicationExitCode
    }
    if ($gpuRestarts -ge $MaximumGpuRestarts) {
        Write-Error 'GPU recovery restart cap reached; refusing a restart loop.'
        exit 76
    }
    $gpuRestarts++
    $env:COLLIDE_O_SCOPE_SUPERVISED_GPU_RESTART = '1'
    Write-Warning 'GPU loss requested a supervised restart. The recovery journal remains operator-owned; no show or audience output will be armed automatically.'
}
