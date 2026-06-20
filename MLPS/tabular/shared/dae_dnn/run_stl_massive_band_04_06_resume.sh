#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

RUN_ROOT="${RUN_ROOT:-MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06}"
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" && -x "./.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="./.venv/Scripts/python.exe"
fi

export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="none"
unset PYTORCH_CUDA_ALLOC_CONF
export TABULAR_CPU_WORKERS=0

exec "$PYTHON_BIN" MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root "$RUN_ROOT" \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 4 6 \
  --repeat-count 5 \
  --scheduler pressure_aware \
  --host-ram-pressure-limit-pct 95 \
  --host-ram-resume-pct 90 \
  --gpu-memory-pressure-limit-pct 85 \
  --gpu-memory-resume-pct 80 \
  --gpu-device-index 0 \
  --max-active-jobs 0 \
  --pressure-poll-interval-sec 0.5 \
  --post-launch-sample-delay-sec 30 \
  --max-epochs 100000000 \
  --num-workers 0 \
  --batch-size 0 \
  --max-width 10000000000 \
  --max-neurons 10000000000 \
  "$@"
