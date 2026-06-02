#!/usr/bin/env bash
set -euo pipefail

cd /home/anurag/Projects/Breaking-the-Neural-Barrier

git checkout main
git pull origin main

concurrency=4
depths=(5 6 7 8)

for ((i=0; i<${#depths[@]}; i+=concurrency)); do
  pids=()
  for ((j=i; j<i+concurrency && j<${#depths[@]}; j++)); do
    depth="${depths[j]}"
    PYTHONUNBUFFERED=1 .venv/bin/python -u MLPS/tabular/shared/dae_dnn/run_with_watchdog.py \
      --run-root MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/generation_d${depth} \
      --idle-seconds 300 \
      --max-restarts 5 \
      --burst-limit 3 \
      --burst-window-seconds 600 \
      --poll-seconds 10 \
      --grace-seconds 20 \
      -- \
      .venv/bin/python -u MLPS/tabular/shared/dae_dnn/run_goliath_staged_width_only.py \
        --data-dir ./data \
        --results-dir MLPS/tabular/shared/dae_dnn/results \
        --run-root MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/generation_d${depth} \
        --tasks generation \
        --stl-depth "${depth}" \
        --alt-start-width 1 \
        --patience 10 \
        --width-expansion-patience 10 \
        --num-workers 0 &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done
