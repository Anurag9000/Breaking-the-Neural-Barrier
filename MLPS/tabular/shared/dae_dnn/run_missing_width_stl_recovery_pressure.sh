#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

# CPU-only default for this recovery family.
# The mixed GPU+CPU launcher lives in
# MLPS/tabular/shared/dae_dnn/run_missing_width_stl_recovery_pressure_gpu_cpu.sh
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="none"
unset PYTORCH_CUDA_ALLOC_CONF

source MLPS/tabular/shared/dae_dnn/runtime_tuning.sh
CPU_CORES="$(tabular_runtime_detect_cpu_cores)"

# CPU performance tuning for the default runner.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_CORES}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_CORES}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_CORES}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_CORES}}"

tabular_runtime_bootstrap
PYTHON_BIN="$(tabular_runtime_python)"

CUDA_VISIBLE_DEVICES="" \
NVIDIA_VISIBLE_DEVICES="none" \
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
  --max-width 10000000000 \
  --max-depth 10 \
  --max-neurons 10000000 \
  --width-stage-margin-patience 10 \
  --width-stage-min-improve-pct 1.0 \
  --repeat-count 5 \
  --width-depths 1,2,3,4,5,6 \
  --missing-present-task-repeats 2,3,4,5 \
  --prediction-repeats 1,2,3,4,5 \
  --host-ram-pressure-limit-pct 85 \
  --host-ram-resume-pct 80 \
  --gpu-memory-pressure-limit-pct 85 \
  --gpu-memory-resume-pct 80 \
  --swap-pressure-limit-pct 100 \
  --swap-resume-pct 100 \
  --gpu-device-index 0 \
  --pressure-poll-interval-sec 0.5 \
  --post-launch-sample-delay-sec 30 \
  --max-active-jobs 0 \
  "$@"
