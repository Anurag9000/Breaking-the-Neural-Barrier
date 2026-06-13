#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_adp_explicit_plan_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/adp/w2d/denoising_plus_sim5_v1 \
  --plan-file MLPS/tabular/shared/dae_dnn/adp_denoising_simulation_suite_plan.json \
  --concurrency 2 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 9312 \
  --adp-mode width_to_depth
