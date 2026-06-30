#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

RUN_ROOT="${RUN_ROOT:-MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow01_03}"
CONCURRENCY=50
MAX_ACTIVE_JOBS=50

source MLPS/tabular/shared/dae_dnn/runtime_tuning.sh
tabular_runtime_bootstrap
PYTHON_BIN="$(tabular_runtime_python)"

CUDA_VISIBLE_DEVICES="" \
"${PYTHON_BIN}" MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root "${RUN_ROOT}" \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 1 3 \
  --repeat-count 5 \
  --scheduler pressure_aware \
  --host-ram-pressure-limit-pct 85 \
  --host-ram-resume-pct 80 \
  --gpu-memory-pressure-limit-pct 90 \
  --gpu-memory-resume-pct 85 \
  --gpu-device-index 0 \
  --max-active-jobs "${MAX_ACTIVE_JOBS}" \
  --max-active-gpu-jobs 0 \
  --concurrency "${CONCURRENCY}" \
  --pressure-poll-interval-sec 0.5 \
  --post-launch-sample-delay-sec 30 \
  --max-epochs 100000000 \
  --num-workers 0 \
  --batch-size 186240 \
  "$@"
