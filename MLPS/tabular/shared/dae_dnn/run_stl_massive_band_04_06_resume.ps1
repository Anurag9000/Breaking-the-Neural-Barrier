$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
Set-Location $RepoRoot

$RunRoot = $env:RUN_ROOT
if (-not $RunRoot) {
    $RunRoot = "MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06"
}

$Python = Join-Path $RepoRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) {
    $Python = Join-Path $RepoRoot ".venv/bin/python"
}
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$env:CUDA_VISIBLE_DEVICES = ""
$env:NVIDIA_VISIBLE_DEVICES = "none"
if (Test-Path env:PYTORCH_CUDA_ALLOC_CONF) {
    Remove-Item env:PYTORCH_CUDA_ALLOC_CONF
}
$env:TABULAR_CPU_WORKERS = "0"

& $Python "MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py" `
  --data-dir ./data `
  --results-dir MLPS/tabular/shared/dae_dnn/results `
  --run-root $RunRoot `
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current `
  --tasks classification autoencoding generation denoising anomaly simulation prediction `
  --param-band 4 6 `
  --repeat-count 5 `
  --scheduler pressure_aware `
  --host-ram-pressure-limit-pct 95 `
  --host-ram-resume-pct 90 `
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
