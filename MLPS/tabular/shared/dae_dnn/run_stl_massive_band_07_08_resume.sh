#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

RUN_ROOT="${RUN_ROOT:-MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow07_08}"
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" && -x "./.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="./.venv/Scripts/python.exe"
fi

export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="none"
unset PYTORCH_CUDA_ALLOC_CONF
export TABULAR_CPU_WORKERS=0
MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-20}"
CONCURRENCY="${CONCURRENCY:-20}"

exec "$PYTHON_BIN" MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root "$RUN_ROOT" \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 7 8 \
  --repeat-count 5 \
  --scheduler pressure_aware \
  --max-active-jobs "$MAX_ACTIVE_JOBS" \
  --concurrency "$CONCURRENCY" \
  --host-ram-pressure-limit-pct 85 \
  --host-ram-resume-pct 80 \
  --gpu-memory-pressure-limit-pct 85 \
  --gpu-memory-resume-pct 80 \
  --pressure-poll-interval-sec 0.5 \
  --post-launch-sample-delay-sec 30 \
  "$@"
