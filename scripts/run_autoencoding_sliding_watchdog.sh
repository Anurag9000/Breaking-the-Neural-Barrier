#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

tasks=(autoencoding)
depths=(1 2 3 4 5 6)
max_parallel=2

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_one() {
  local task="$1"
  local depth="$2"

  .venv/bin/python MLPS/tabular/shared/dae_dnn/run_with_watchdog.py \
    --run-root MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/${task}_d${depth} \
    --idle-seconds 120 \
    --max-restarts 5 \
    --burst-limit 3 \
    --burst-window-seconds 600 \
    --poll-seconds 10 \
    --grace-seconds 20 \
    -- \
    .venv/bin/python MLPS/tabular/shared/dae_dnn/run_goliath_staged_width_only.py \
      --data-dir ./data \
      --results-dir MLPS/tabular/shared/dae_dnn/results \
      --run-root MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/${task}_d${depth} \
      --tasks "$task" \
      --stl-depth "$depth" \
      --alt-start-width 1 \
      --patience 10 \
      --width-expansion-patience 10 \
      --num-workers 0
}

active_pids=()
active_labels=()

cleanup() {
  local pid
  for pid in "${active_pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait || true
}

trap cleanup INT TERM

start_job() {
  local task="$1"
  local depth="$2"
  log "Starting ${task} d${depth}"
  run_one "$task" "$depth" &
  active_pids+=("$!")
  active_labels+=("${task}_d${depth}")
}

prune_finished_jobs() {
  local -a next_pids=()
  local -a next_labels=()
  local i pid label

  for i in "${!active_pids[@]}"; do
    pid="${active_pids[$i]}"
    label="${active_labels[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      next_pids+=("$pid")
      next_labels+=("$label")
    else
      log "Finished ${label}"
    fi
  done

  active_pids=("${next_pids[@]}")
  active_labels=("${next_labels[@]}")
}

for task in "${tasks[@]}"; do
  next_depth_idx=0
  while (( next_depth_idx < ${#depths[@]} || ${#active_pids[@]} > 0 )); do
    while (( next_depth_idx < ${#depths[@]} && ${#active_pids[@]} < max_parallel )); do
      start_job "$task" "${depths[$next_depth_idx]}"
      next_depth_idx=$((next_depth_idx + 1))
    done

    if (( ${#active_pids[@]} == 0 )); then
      break
    fi

    wait -n || true
    prune_finished_jobs
  done
done
