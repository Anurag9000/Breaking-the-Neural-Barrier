#!/usr/bin/env bash
set -euo pipefail

cd /home/anurag/Projects/Breaking-the-Neural-Barrier

git checkout main
git pull origin main

concurrency=3
depths=(1 2 3 4 5 6 7 8 9 10)

launch_depth() {
  local depth="$1"
  echo "[run_generation_suite] launching depth ${depth}"
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
}

active=0
next_index=0
total="${#depths[@]}"

while (( next_index < total )); do
  while (( active < concurrency && next_index < total )); do
    launch_depth "${depths[next_index]}"
    active=$((active + 1))
    next_index=$((next_index + 1))
  done

  if (( active > 0 )); then
    set +e
    wait -n
    status=$?
    set -e
    echo "[run_generation_suite] one depth finished (exit ${status})"
    active=$((active - 1))
  fi
done

while (( active > 0 )); do
  set +e
  wait -n
  status=$?
  set -e
  echo "[run_generation_suite] one depth finished (exit ${status})"
  active=$((active - 1))
done
