#!/usr/bin/env bash
# run_missing_width_stl_recovery_pressure_gpu_first.sh
# GPU-first dual-gate recovery runner: fills GPU VRAM first, falls back to CPU RAM.
# GPU gate reopens only on GPU job completion or >500 MiB GPU VRAM drop.
# CPU gate follows normal RAM gating logic.
# Swap pressure monitoring also active (default thresholds allow full swap use).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/runtime_tuning.sh"
tabular_runtime_bootstrap
PYTHON_BIN="$(tabular_runtime_python)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export TABULAR_CPU_WORKERS="${TABULAR_CPU_WORKERS:-0}"

exec "$PYTHON_BIN" \
  MLPS/tabular/shared/dae_dnn/run_missing_width_stl_recovery_pressure.py \
  --scheduler gpu_first \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/recovery/missing_width_stl_v1 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --batch-size 0 \
  --num-workers 0 \
  --seed 0 \
  --repeat-count 5 \
  --width-depths 1,2,3,4,5,6 \
  --missing-present-task-repeats 2,3,4,5 \
  --prediction-repeats 1,2,3,4,5 \
  --max-active-jobs 0 \
  --max-active-gpu-jobs 0 \
  --gpu-device-index 0 \
  --host-ram-pressure-limit-pct 90.0 \
  --host-ram-resume-pct 85.0 \
  --gpu-memory-pressure-limit-pct 90.0 \
  --gpu-memory-resume-pct 85.0 \
  --swap-pressure-limit-pct 100.0 \
  --swap-resume-pct 100.0 \
  --pressure-poll-interval-sec 0.5 \
  --post-launch-sample-delay-sec 30.0 \
  --batch-backoff-factor 0.5 \
  "$@"
