#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

# Mixed GPU+CPU recovery runner.
# It shares the same run root as the CPU default runner so checkpoints and
# candidate state can move between CPU and GPU launches without any layout fork.
source MLPS/tabular/shared/dae_dnn/runtime_tuning.sh
tabular_runtime_bootstrap
PYTHON_BIN="$(tabular_runtime_python)"

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON_BIN}" MLPS/tabular/shared/dae_dnn/run_missing_width_stl_recovery_pressure.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/recovery/missing_width_stl_v1 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --batch-size 186240 \
  --num-workers 0 \
  --seed 0 \
  --patience 10 \
  --delta 1e-4 \
  --max-epochs 100000000 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --max-width 4096 \
  --max-depth 10 \
  --max-neurons 10000000 \
  --width-stage-margin-patience 10 \
  --width-stage-min-improve-pct 1.0 \
  --repeat-count 5 \
  --width-depths 1,2,3,4,5,6 \
  --missing-present-task-repeats 2,3,4,5 \
  --prediction-repeats 1,2,3,4,5 \
  --host-ram-pressure-limit-pct 90 \
  --host-ram-resume-pct 85 \
  --gpu-memory-pressure-limit-pct 90 \
  --gpu-memory-resume-pct 85 \
  --swap-pressure-limit-pct 100 \
  --swap-resume-pct 100 \
  --gpu-device-index 0 \
  --pressure-poll-interval-sec 0.5 \
  --post-launch-sample-delay-sec 60 \
  --max-active-jobs 0 \
  "$@"
