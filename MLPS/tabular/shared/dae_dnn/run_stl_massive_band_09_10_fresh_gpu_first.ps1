# run_stl_massive_band_09_10_fresh_gpu_first.ps1
# GPU-first dual-gate scheduler for parameter-decade band 10^9-10^10 (fresh run).
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir "..\..\..\..")).Path
Set-Location $RepoRoot

$CoreCount = [Environment]::ProcessorCount
if (-not $env:OMP_NUM_THREADS)          { $env:OMP_NUM_THREADS          = $CoreCount }
if (-not $env:MKL_NUM_THREADS)          { $env:MKL_NUM_THREADS          = $CoreCount }
if (-not $env:OPENBLAS_NUM_THREADS)     { $env:OPENBLAS_NUM_THREADS     = $CoreCount }
if (-not $env:NUMEXPR_NUM_THREADS)      { $env:NUMEXPR_NUM_THREADS      = $CoreCount }
if (-not $env:VECLIB_MAXIMUM_THREADS)   { $env:VECLIB_MAXIMUM_THREADS   = $CoreCount }
if (-not $env:TORCH_NUM_THREADS)        { $env:TORCH_NUM_THREADS        = $CoreCount }
$env:TORCH_INTEROP_THREADS = "1"
$env:OMP_DYNAMIC = "FALSE"
$env:MKL_DYNAMIC = "FALSE"
$env:OMP_WAIT_POLICY = "ACTIVE"
$env:TABULAR_CHILD_SHARED_CPU = "1"
$env:TABULAR_CPU_WORKERS = "0"

if (-not $env:CUDA_VISIBLE_DEVICES)    { $env:CUDA_VISIBLE_DEVICES    = "0" }
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128" }

$PythonBin = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" }
             elseif (Test-Path ".venv/bin/python")      { ".venv/bin/python" }
             else                                        { "python" }

& $PythonBin `
  MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py `
  --scheduler gpu_first `
  --tasks classification autoencoding generation denoising anomaly simulation prediction `
  --param-band 9 10 `
  --data-dir ./data `
  --results-dir MLPS/tabular/shared/dae_dnn/results `
  --run-root MLPS/tabular/shared/dae_dnn/results/parammatched_decade_v1_fresh_v1_gpu_first_param_10pow09_10 `
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current `
  --batch-size 0 --num-workers 0 --seed 0 --repeat-count 5 `
  --max-active-jobs 0 --max-active-gpu-jobs 0 --gpu-device-index 0 `
  --host-ram-pressure-limit-pct 90.0 --host-ram-resume-pct 85.0 `
  --gpu-memory-pressure-limit-pct 90.0 --gpu-memory-resume-pct 85.0 `
  --pressure-poll-interval-sec 0.5 --post-launch-sample-delay-sec 30.0 `
  --batch-backoff-factor 0.5 `
  @args
