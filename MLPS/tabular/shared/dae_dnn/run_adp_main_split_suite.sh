#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

source MLPS/tabular/shared/dae_dnn/runtime_tuning.sh
tabular_runtime_bootstrap
PYTHON_BIN="$(tabular_runtime_python)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" MLPS/tabular/shared/dae_dnn/run_adp_explicit_plan_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/adp/w2d/main_without_denoising_v1 \
  --plan-file MLPS/tabular/shared/dae_dnn/adp_main_suite_plan.json \
  --concurrency 6 \
  --num-workers 0 \
  --batch-size 93120 \
  --adp-mode width_to_depth
