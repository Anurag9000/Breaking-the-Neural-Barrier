#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

RUN_ROOT="${RUN_ROOT:-MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow07_08_fresh_v1}"
source MLPS/tabular/shared/dae_dnn/runtime_tuning.sh
tabular_runtime_bootstrap
PYTHON_BIN="$(tabular_runtime_python)"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON_BIN}" MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root "${RUN_ROOT}" \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 7 8 \
  --repeat-count 5 \
  --scheduler pressure_aware \
  --host-ram-pressure-limit-pct 90 \
  --host-ram-resume-pct 85 \
  --gpu-memory-pressure-limit-pct 90 \
  --gpu-memory-resume-pct 85 \
  --gpu-device-index 0 \
  --max-active-jobs 0 \
  --pressure-poll-interval-sec 0.5 \
  --post-launch-sample-delay-sec 60 \
  --max-epochs 100000000 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 186240 \
  "$@"
