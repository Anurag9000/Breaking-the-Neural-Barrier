param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
$originalRunner = Join-Path $PSScriptRoot "run_stl_massive_band_07_10_smart_recovery_original.ps1"
$thermalGuard = Join-Path $repoRoot "thermal_guard_embedded.ps1"
$stopFile = Join-Path $repoRoot "thermal_guard.stop"

if (Test-Path $stopFile) {
    Remove-Item $stopFile -Force -ErrorAction SilentlyContinue
}

$previousSchemeLine = powercfg /getactivescheme
$previousScheme = [regex]::Match($previousSchemeLine, "[0-9a-fA-F-]{36}").Value

Write-Host "============================================================"
Write-Host " AUTO THERMAL GUARD + PERFORMANCE MODE"
Write-Host "============================================================"
Write-Host "Previous power scheme: $previousScheme"
Write-Host "Starting thermal guard..."
Write-Host "Logs: thermal_guard.log"
Write-Host "============================================================"

$guardProc = Start-Process powershell -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$thermalGuard`"",
    "-StopFile", "`"$stopFile`"",
    "-HotTempC", "90",
    "-CoolTempC", "85",
    "-HotCpuMax", "50",
    "-NormalCpuMax", "100",
    "-PollSeconds", "5"
) -PassThru -WindowStyle Minimized

try {
    & $originalRunner @RunnerArgs
    $exitCode = $LASTEXITCODE
    if ($null -ne $exitCode -and $exitCode -ne 0) {
        exit $exitCode
    }
}
finally {
    Write-Host "Stopping thermal guard..."
    New-Item -ItemType File -Path $stopFile -Force | Out-Null

    Start-Sleep -Seconds 8

    if ($guardProc -and -not $guardProc.HasExited) {
        Stop-Process -Id $guardProc.Id -Force -ErrorAction SilentlyContinue
    }

    if ($previousScheme) {
        Write-Host "Restoring previous power scheme: $previousScheme"
        powercfg /s $previousScheme | Out-Null
    }

    Remove-Item $stopFile -Force -ErrorAction SilentlyContinue

    Write-Host "Thermal guard stopped. Previous power scheme restored."
}
