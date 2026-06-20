#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

RUN_ROOT="${RUN_ROOT:-MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow07_08}"
MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-2}"
CONCURRENCY="${CONCURRENCY:-2}"

# Strict CPU-only: hide GPU before any runtime bootstrap/code can see it.
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="none"
unset PYTORCH_CUDA_ALLOC_CONF

# CPU performance tuning: start with all logical CPUs and let the runtime
# bootstrap preserve these ceilings while it sets affinity and priority.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$(nproc)}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$(nproc)}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$(nproc)}"

source MLPS/tabular/shared/dae_dnn/runtime_tuning.sh
tabular_runtime_bootstrap
PYTHON_BIN="$(tabular_runtime_python)"

export TABULAR_SKIP_SYSTEMD_SCOPE="${TABULAR_SKIP_SYSTEMD_SCOPE:-0}"

CUDA_VISIBLE_DEVICES="" \
NVIDIA_VISIBLE_DEVICES="none" \
"${PYTHON_BIN}" MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root "${RUN_ROOT}" \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 7 8 \
  --repeat-count 5 \
  --scheduler pressure_aware \
  --host-ram-pressure-limit-pct 85 \
  --host-ram-resume-pct 80 \
  --gpu-memory-pressure-limit-pct 85 \
  --gpu-memory-resume-pct 80 \
  --gpu-device-index 0 \
  --max-active-gpu-jobs 0 \
  --concurrency "${CONCURRENCY}" \
  --max-active-jobs "${MAX_ACTIVE_JOBS}" \
  --pressure-poll-interval-sec 0.5 \
  --post-launch-sample-delay-sec 30 \
  --max-epochs 100000000 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 186240 \
  "$@"
