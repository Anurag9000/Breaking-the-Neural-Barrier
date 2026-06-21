$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
Set-Location $RepoRoot

$CpuCores = [Environment]::ProcessorCount
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "$CpuCores" }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = "$CpuCores" }
if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = "$CpuCores" }
if (-not $env:NUMEXPR_NUM_THREADS) { $env:NUMEXPR_NUM_THREADS = "$CpuCores" }
if (-not $env:VECLIB_MAXIMUM_THREADS) { $env:VECLIB_MAXIMUM_THREADS = "$CpuCores" }
if (-not $env:TORCH_NUM_THREADS) { $env:TORCH_NUM_THREADS = "$CpuCores" }
if (-not $env:TORCH_INTEROP_THREADS) { $env:TORCH_INTEROP_THREADS = "1" }
if (-not $env:OMP_DYNAMIC) { $env:OMP_DYNAMIC = "FALSE" }
if (-not $env:MKL_DYNAMIC) { $env:MKL_DYNAMIC = "FALSE" }
if (-not $env:OMP_WAIT_POLICY) { $env:OMP_WAIT_POLICY = "ACTIVE" }
if (-not $env:TABULAR_CHILD_SHARED_CPU) { $env:TABULAR_CHILD_SHARED_CPU = "1" }
if (-not $env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES = "0" }
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"
}

$Python = Join-Path $RepoRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) {
    $Python = Join-Path $RepoRoot ".venv/bin/python"
}
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " [STL MASSIVE] BAND 7-8 SMART RECOVERY & BAND 9-10 EXECUTION " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Phase 1: Band 7-8 Smart Recovery
# By pointing to the original 07_08_fresh_v1 directory, the parallel orchestrator
# natively parses ablation_state.json and summary.json to detect completed repeats and skip them.
# Incomplete or abruptly stopped runs lack a final state, so they will start afresh.
Write-Host "`n>>> PHASE 1: Band 7-8 Smart Recovery" -ForegroundColor Yellow
$RunRoot78 = "MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow07_08_fresh_v1"

& $Python "MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py" `
  --data-dir ./data `
  --results-dir MLPS/tabular/shared/dae_dnn/results `
  --run-root $RunRoot78 `
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current `
  --tasks classification autoencoding generation denoising anomaly simulation prediction `
  --param-band 7 8 `
  --repeat-count 5 `
  --scheduler pressure_aware `
  --host-ram-pressure-limit-pct 85 `
  --host-ram-resume-pct 80 `
  --gpu-memory-pressure-limit-pct 85 `
  --gpu-memory-resume-pct 80 `
  --gpu-device-index 0 `
  --max-active-jobs 0 `
  --pressure-poll-interval-sec 0.5 `
  --post-launch-sample-delay-sec 30 `
  --max-epochs 100000000 `
  --num-workers 0 `
  --batch-size 0 `
  --max-width 10000000000 `
  --max-neurons 10000000000 `
  @args

if ($LASTEXITCODE -ne 0) { Write-Error "Phase 1 Failed"; exit $LASTEXITCODE }

# Phase 2: Band 9-10 Execution
Write-Host "`n>>> PHASE 2: Band 9-10 Fresh Execution" -ForegroundColor Yellow
$RunRoot910 = "MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow09_10_fresh_v1"

& $Python "MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py" `
  --data-dir ./data `
  --results-dir MLPS/tabular/shared/dae_dnn/results `
  --run-root $RunRoot910 `
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current `
  --tasks classification autoencoding generation denoising anomaly simulation prediction `
  --param-band 9 10 `
  --repeat-count 5 `
  --scheduler pressure_aware `
  --host-ram-pressure-limit-pct 85 `
  --host-ram-resume-pct 80 `
  --gpu-memory-pressure-limit-pct 85 `
  --gpu-memory-resume-pct 80 `
  --gpu-device-index 0 `
  --max-active-jobs 0 `
  --pressure-poll-interval-sec 0.5 `
  --post-launch-sample-delay-sec 30 `
  --max-epochs 100000000 `
  --num-workers 0 `
  --batch-size 0 `
  --max-width 10000000000 `
  --max-neurons 10000000000 `
  @args

if ($LASTEXITCODE -ne 0) { Write-Error "Phase 2 Failed"; exit $LASTEXITCODE }

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " [STL MASSIVE] SUITE COMPLETED SUCCESSFULLY " -ForegroundColor Green
Write-Host " Note: Band 7-10 outputs are safely preserved in their respective directories for later merging." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
