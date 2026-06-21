$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) {
    $Python = Join-Path $RepoRoot ".venv/bin/python"
}
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Triggering cross-platform MLPS emergency kill switch..." -ForegroundColor Yellow
& $Python "scripts/kill_all_runners.py"
