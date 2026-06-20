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

$RunRoot = $env:RUN_ROOT
if (-not $RunRoot) {
    $RunRoot = "MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06_fresh_v1"
}

$Python = Join-Path $RepoRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) {
    $Python = Join-Path $RepoRoot ".venv/bin/python"
}
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python "MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py" `
  --data-dir ./data `
  --results-dir MLPS/tabular/shared/dae_dnn/results `
  --run-root $RunRoot `
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current `
  --tasks classification autoencoding generation denoising anomaly simulation prediction `
  --param-band 4 6 `
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
  --pin-memory `
  --batch-size 186240 `
  @args
