$ErrorActionPreference = "Stop"

function Test-Admin {
    return ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

if (-not (Test-Admin)) {
    Write-Host "Re-launching FULL GUARDED RUNNER as Administrator..."
    Start-Process powershell -Verb RunAs -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`""
    )
    exit
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
Set-Location $repoRoot

$originalRunner = Join-Path $PSScriptRoot "run_stl_massive_band_07_10_smart_recovery_original.ps1"
$thermalGuard = Join-Path $repoRoot "thermal_guard_embedded.ps1"
$strongAwakeGuard = Join-Path $repoRoot "strong_awake_guard.ps1"
$stopFile = Join-Path $repoRoot "thermal_guard.stop"

if (!(Test-Path $originalRunner)) { throw "Missing original runner: $originalRunner" }
if (!(Test-Path $thermalGuard)) { throw "Missing thermal guard: $thermalGuard" }
if (!(Test-Path $strongAwakeGuard)) { throw "Missing strong awake guard: $strongAwakeGuard" }

if (Test-Path $stopFile) {
    Remove-Item $stopFile -Force -ErrorAction SilentlyContinue
}

$previousSchemeLine = powercfg /getactivescheme
$previousScheme = [regex]::Match($previousSchemeLine, "[0-9a-fA-F-]{36}").Value

Write-Host "============================================================"
Write-Host " FULL GUARDED RUNNER"
Write-Host "============================================================"
Write-Host "Admin mode: YES"
Write-Host "Previous power scheme: $previousScheme"
Write-Host "Repo root: $repoRoot"
Write-Host "Starting strong awake guard + thermal guard..."
Write-Host "============================================================"

$env:PYTHONPATH = $repoRoot.Path

$awakeProc = $null
$thermalProc = $null

try {
    $awakeProc = Start-Process powershell -WorkingDirectory $repoRoot -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$strongAwakeGuard`""
    ) -PassThru -WindowStyle Minimized

    Start-Sleep -Seconds 2

    $thermalProc = Start-Process powershell -WorkingDirectory $repoRoot -ArgumentList @(
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

    Start-Sleep -Seconds 5

    Write-Host "Power requests currently active:"
    powercfg /requests

    Write-Host "============================================================"
    Write-Host "Starting original experiment runner..."
    Write-Host "============================================================"

    & $originalRunner

    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Write-Host "Stopping thermal guard..."
    New-Item -ItemType File -Path $stopFile -Force | Out-Null
    Start-Sleep -Seconds 8

    if ($thermalProc -and -not $thermalProc.HasExited) {
        Stop-Process -Id $thermalProc.Id -Force -ErrorAction SilentlyContinue
    }

    Write-Host "Stopping strong awake guard..."
    if ($awakeProc -and -not $awakeProc.HasExited) {
        Stop-Process -Id $awakeProc.Id -Force -ErrorAction SilentlyContinue
    }

    if ($previousScheme) {
        Write-Host "Restoring previous power scheme: $previousScheme"
        powercfg /s $previousScheme | Out-Null
    }

    Remove-Item $stopFile -Force -ErrorAction SilentlyContinue

    Write-Host "Full guarded runner stopped. Previous power plan restored."
}

