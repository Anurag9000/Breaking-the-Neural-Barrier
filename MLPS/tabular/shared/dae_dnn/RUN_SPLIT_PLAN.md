# Split Run Plan

This file is the operator handoff for the current width-only sweep.
It is written so another Codex agent can pick it up without guessing.

## Goal

Run the active tabular tasks in depth waves:

- wave 1: depths `1 2 3 4 5`
- wave 2: depths `6 7 8`
- wave 3: depths `9 10`

Run each task independently on its own machine when possible.
Each per-depth run root is resumable.

## Current task assignments

### This laptop

Task:

- `classification`

Execution order:

1. depths `1 2 3 4 5`
2. depths `6 7 8`
3. depths `9 10`

Use the same run root prefix for all three waves:

`MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu`

### Other laptop 1

Task:

- `autoencoding`

Execution order:

1. depths `1 2 3 4 5`
2. depths `6 7 8`
3. depths `9 10`

### Other laptop 2

Task:

- `generation`

Execution order:

1. depths `1 2 3 4 5`
2. depths `6 7 8`
3. depths `9 10`

### Other laptop 3

Task:

- `denoising`

Execution order:

1. depths `1 2 3 4 5`
2. depths `6 7 8`
3. depths `9 10`

## Exact command template

Use this command template for every wave:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier
git checkout main
git pull origin main

.venv/bin/python MLPS/tabular/shared/dae_dnn/run_goliath_staged_width_only.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/<task>_d<depth> \
  --tasks <task> \
  --stl-depth <depth> \
  --alt-start-width 1 \
  --patience 10 \
  --width-expansion-patience 10 \
  --num-workers 0
```

For a wave, launch the four or two depth jobs in parallel, then wait for them to finish.

## This laptop: classification wave commands

### Wave 1

Depths `1 2 3 4 5`:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier
git checkout main
git pull origin main

tasks=(classification)
for task in "${tasks[@]}"; do
  for depth in 1 2 3 4 5; do
    .venv/bin/python MLPS/tabular/shared/dae_dnn/run_goliath_staged_width_only.py \
      --data-dir ./data \
      --results-dir MLPS/tabular/shared/dae_dnn/results \
      --run-root MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/${task}_d${depth} \
      --tasks "$task" \
      --stl-depth "$depth" \
      --alt-start-width 1 \
      --patience 10 \
      --width-expansion-patience 10 \
      --num-workers 0 &
  done
  wait
done
```

### Wave 2

Depths `6 7 8`:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier
git checkout main
git pull origin main

tasks=(classification)
for task in "${tasks[@]}"; do
  for depth in 6 7 8; do
    .venv/bin/python MLPS/tabular/shared/dae_dnn/run_goliath_staged_width_only.py \
      --data-dir ./data \
      --results-dir MLPS/tabular/shared/dae_dnn/results \
      --run-root MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/${task}_d${depth} \
      --tasks "$task" \
      --stl-depth "$depth" \
      --alt-start-width 1 \
      --patience 10 \
      --width-expansion-patience 10 \
      --num-workers 0 &
  done
  wait
done
```

### Wave 3

Depths `9 10`:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier
git checkout main
git pull origin main

tasks=(classification)
for task in "${tasks[@]}"; do
  for depth in 9 10; do
    .venv/bin/python MLPS/tabular/shared/dae_dnn/run_goliath_staged_width_only.py \
      --data-dir ./data \
      --results-dir MLPS/tabular/shared/dae_dnn/results \
      --run-root MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/${task}_d${depth} \
      --tasks "$task" \
      --stl-depth "$depth" \
      --alt-start-width 1 \
      --patience 10 \
      --width-expansion-patience 10 \
      --num-workers 0 &
  done
  wait
done
```

## Resume rule

If a run stops, rerun the same wave command with the same `--run-root`.
Do not change the per-depth run root.

## Consolidation rule

After a wave finishes on any machine:

1. commit the updated run-root artifacts
2. push to `main`
3. the next machine pulls `main`
4. continue with the next wave or the next task

## Tasks to keep in GitHub as archived context

These are already preserved in the repo state and should stay visible to future agents:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`
- STL ablations for all tasks
- archived width-to-depth and STL comparison artifacts
