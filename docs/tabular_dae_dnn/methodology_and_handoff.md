# Tabular DAE/DNN Methodology and Handoff

This document records the exact code paths and workflow used for the tabular
MLP experiments in this repo. Use it as the reference when running the same
study on another machine.

The results layout described here is the only accepted structure for future
MLPS tabular DAE/DNN outputs in this repo, and the final state should be
published on `main` only.

## Canonical task names

Current task names:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`

## Exact code paths to use

Task definitions and dataset mapping:

- `MLPS/tabular/shared/dae_dnn/tasks.py`

STL sweep orchestration:

- `MLPS/tabular/shared/dae_dnn/run_stl_ablation.py`
- `MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py`
- `MLPS/tabular/shared/dae_dnn/run_stl_parallelism_probe.py`
- `MLPS/tabular/shared/dae_dnn/generate_stl_planned_params_report.py`
- `MLPS/tabular/shared/dae_dnn/run_stl_small_grid.py`

ADP width-to-depth orchestration:

- `MLPS/tabular/shared/dae_dnn/run_task.py`
- `MLPS/tabular/shared/dae_dnn/adp_search.py`
- `utils/adp_contract.py`
- `MLPS/tabular/shared/dae_dnn/run_adp_w2d_suite_parallel.py`

Historical ADP W2D archives already restored into the repo:

- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_anomaly_onward_gpu`
- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_staged_current`

The staged legacy lineage is exposed in the repo-visible archive root as
`classification` for that task lineage.

## ADP Resume Order

Use `run_adp_w2d_suite_parallel.py` for resumable ADP width-to-depth suites.
It works repeat by repeat:

- inside a repeat, completed task roots are skipped
- incomplete task roots are resumed from their saved task root
- new task roots in that same repeat are started up to `--concurrency`
- the launcher does not open the next repeat until the current repeat is
  fully finished

For the current `repeat4_plus_sim_pred5_v1` suite, the saved snapshot has:

- `repeat_01/denoising` incomplete
- all other `repeat_01` tasks complete

So the next resume starts with `repeat_01/denoising` only. After that repeat
finishes, the launcher opens `repeat_02` and starts all seven tasks there.

## Width-to-depth rule

For width-to-depth runs, a depth expansion is not treated as a terminal step.
After the new layer is added, the controller continues widening neuron by
neuron until the model is uniform again, and only then does the patience
counter resume. The width-stage failure and margin counters are reset after a
completed depth/fill cycle, so a model like `[29, 29, 29]` can continue to
`[30, 29, 29]`, `[30, 30, 29]`, and `[30, 30, 30]` before width patience is
allowed to stop the search. That prevents the run from quitting while the
post-depth warmup is still in progress.

## Small historical STL archive

This is separate from the massive STL ablation TODO.

The lightweight historical STL archive at `results/archive/classification_trial1`
is a one-off fixed-grid summary, not a repeat suite. It covers the following
tasks:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`

The archived STL ablation grid is sparse and fixed:

- depths: `3`, `4`, `6`, `8`, `10`
- widths: `64`, `96`, `128`, `160`, `192`, `224`, `256`
- repeats: none

The repo-facing task layout for this archive is task-first. Each task root is
organized as:

- `w2d/`
- `width_only/`
- `stl_ablation/`

Within `stl_ablation/`, depths are the outer folders and widths are the
nested subfolders. Within `width_only/`, depth folders are the outer layer
and the width candidates are nested below that.

That exact nesting is the required structure for any future MLPS tabular
result drop:

- task at the top
- `w2d/`, `width_only/`, `stl_ablation/` immediately below
- depth folders inside `width_only/` and `stl_ablation/`
- width folders inside each depth folder
- cross-task rollups under `analysis/`
- publish the final merged tree on `main`

Recovered small-grid coverage today:

- `classification`: STL ablation and W2D recovered; width-only is a placeholder
- `autoencoding`: STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `generation`: STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `denoising`: STL ablation, W2D, and width-only recovered for depths `d1` through `d10`
- `anomaly`: STL ablation and W2D recovered; width-only is a placeholder
- `simulation`: STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only are placeholders
- `prediction`: STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only are placeholders

The `simulation` and `prediction` slave-laptop follow-up has already been
assimilated into the canonical task roots:

- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/simulation`
- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/prediction`

Its suite-level provenance remains under:

- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/analysis/simulation_prediction_v1`

Same `--run-root` means resume. Keep this separate from the massive all-task
STL sweep.

## Small STL follow-up on slave laptops

If you want to rerun the `simulation` and `prediction` small-grid follow-up on
a slave machine, use the dedicated no-repeat grid runner and keep it out of
the massive all-task STL ablation tree.

Use this command:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_small_grid.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/small_grid/simulation_prediction_v1 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/archive/classification_trial1 \
  --tasks simulation prediction \
  --depths 3 4 6 8 10 \
  --widths 64 96 128 160 192 224 256 \
  --num-workers 0 \
  --patience 10 \
  --batch-size 9312 \
  --max-epochs 100000000
```

This follow-up is separate from the massive all-task STL TODO and should not
be mixed with the live repeat-4 ADP run. It is one run per `(task, depth,
width)` candidate, with no repeats.

## Massive STL parameter-band split

The massive STL ablation is still the main long-running study, but it is now
split by parameter-count decade bands so multiple laptops can work on
disjoint slices in parallel and the results can be merged later.

Use the same runner, but cap the parameter-decade range per machine:

- machine 1: parameter decades `1` through `3`
- machine 2: parameter decades `4` through `6`
- machine 3: parameter decades `7` and `8`
- machine 4: parameter decades `9` and `10`

The launcher now accepts `--param-band START END`, where the values are
parameter-count exponents. For example, `--param-band 1 3` keeps targets in
the `10^1` through `10^3` range.

For the real massive STL ablation, training is intended to stop by early
stopping only. There is no intended short epoch cap on the actual ablation
run. The real runner default is effectively unbounded (`--max-epochs
100000000`) so patience is the stopping mechanism. The only intentionally
short run in this workflow is the parallelism probe, which is capped at two
epochs on purpose.

Before starting the real ablation on a given laptop, prefer the
pressure-aware scheduler in
`MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py`. It now expands
the concrete STL child runs for the chosen band, sorts them globally from
smallest to largest parameter count, and opportunistically fills the machine
with as many children as fit. It monitors host RAM usage from `/proc/meminfo`.
If used RAM crosses the configured pressure threshold, it requests a pause on
the largest active child, terminates only that child process group, and
requeues that same child root. When enough RAM is free again, the scheduler
relaunches the paused child from the same run directory, so resumability is
handled by the normal STL checkpoints and `ablation_state.json`.

Relevant flags:

- `--scheduler pressure_aware`
- `--host-ram-pressure-limit-pct 90`
- `--host-ram-resume-pct 85`
- `--max-active-jobs 0`

The old probe path is still available if you want a fixed-slot concurrency
number for a specific laptop. The probe starts at `N=2`, launches the `N`
largest parameter-count candidates first, runs each of them for exactly two
epochs through the normal checkpoint/resume path, and increases `N` until a
trial fails. The last successful `N` is the fixed concurrency to use for the
legacy scheduler.

The probe writes two files into its run root:

- `recommended_parallelism.txt`
- `parallelism_probe_summary.json`

The real launcher accepts `--concurrency-file` and can read the discovered
value directly.

If you want to inspect the full candidate layout before running anything,
generate the per-task plan report first. It writes:

- `planned_params_by_task_depth_width.csv`
- `planned_target_samples_by_task_depth_width.csv`
- per-task `planned_target_samples.csv`
- per-task `planned_candidate_families.csv`
- one plot per task showing the decade distribution and depth coverage

Example planner command:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

./.venv/bin/python MLPS/tabular/shared/dae_dnn/generate_stl_planned_params_report.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --output-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/analysis/planned_params \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 1 3 \
  --num-workers 0 \
  --batch-size 9312
```

The planner reports every sampled target per depth and the deduped candidate
families that the actual run will execute. Use it when you want to see the
exact candidate spread before launching the banded run.

Example optional probe command:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_parallelism_probe.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/parallelism_probe/param_10pow01_03 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 1 3 \
  --probe-epochs 2 \
  --start-n 2 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 9312
```

Example real run command using the pressure-aware scheduler:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow01_03 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 1 3 \
  --repeat-count 5 \
  --scheduler pressure_aware \
  --host-ram-pressure-limit-pct 90 \
  --host-ram-resume-pct 85 \
  --max-active-jobs 0 \
  --max-epochs 100000000 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 9312
```

Example legacy fixed-slot run using the probe output:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow01_03 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 1 3 \
  --concurrency-file MLPS/tabular/shared/dae_dnn/results/stl/parallelism_probe/param_10pow01_03/recommended_parallelism.txt \
  --repeat-count 5 \
  --max-epochs 100000000 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 9312
```

Example for one band:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow01_03 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 1 3 \
  --repeat-count 5 \
  --concurrency 7 \
  --max-epochs 100000000 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 9312
```

Repeat the same command on the other laptops, changing only the parameter band
and the `--run-root` suffix.

The pressure-aware scheduler runs concrete children globally from
smallest-to-largest and may temporarily pause the largest one if host memory
pressure spikes. The probe uses the opposite order and is intentionally
adversarial: it stress tests the heaviest models first.

After all bands finish, merge them into the canonical STL root:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

./.venv/bin/python MLPS/tabular/shared/dae_dnn/merge_stl_ablation_bands.py \
  --input-roots \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow01_03 \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06 \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow07_08 \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow09_10 \
  --output-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1
```

The merge step rebuilds the combined task summaries and plots from the split
band outputs. Keep the band roots as staging inputs until the combined root is
regenerated.

## ADP repeat schedule

The current ADP width-to-depth suite is launcher-driven. Keep the active
`--run-root` aligned with the launcher state and do not reuse a deleted root.
The repeat schedule is encoded in the launcher, not in this document.

Current live ADP repeat split:

- `classification`, `autoencoding`, `generation`, `denoising`, `anomaly`: 4 repeats in the active suite
- `simulation`, `prediction`: 5 repeats in the active suite
- the first five tasks are intended to be merged later with the prior one-off W2D history so each task has 5 combined repeats in the canonical combined tree

The older one-off ADP W2D outputs for the first five tasks are intended to be
merged into the current canonical base later, so the combined history stays
under one task family instead of being split across unrelated roots.

Legacy and analysis helpers:

- `MLPS/tabular/shared/dae_dnn/run_goliath.py`
- `MLPS/tabular/shared/dae_dnn/generate_loss_vs_params_plots.py`
- `MLPS/tabular/shared/dae_dnn/generate_width_only_depth_sweep_plots.py`
- `MLPS/tabular/shared/dae_dnn/generate_w2d_trajectory_loglog_plots.py`
- `MLPS/tabular/shared/dae_dnn/recover_trial1_w2d_history_from_git.py`

## Current training style

The repo uses the same broad method pattern across the tabular sweeps:

1. build the task from `tasks.py`
2. launch a runner entry point
3. write run metadata, logs, CSVs, and phase/task state into the run root
4. restore from the saved checkpoint/state files when resuming
5. keep the repeat-level results separate from the derived analysis outputs
6. probe parameter-band parallelism before launching the real massive STL band

The current result roots are:

- STL: `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- ADP: `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- archive: `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`

## Resume rule

- same `--run-root` means resume
- new `--run-root` means fresh run
- do not change task order, candidate grid, or repeat count while resuming
- keep the local `main` branch authoritative

## Export rule for other laptops

Non-master laptops should export only generated experiment data:

- `results/`
- `runs/`
- `logs/`
- `metrics/`
- `plots/`
- `checkpoints/`
- `tensorboard/`
- CSV and JSON outputs
- generated figures

Do not merge code changes from those machines into canonical `main`.

## Recommended refresh commands

Refresh the inventory after new results land:

```bash
./.venv/bin/python scripts/update_experiment_inventory.py
```

## Result organization

The canonical result layout is documented in:

- `MLPS/tabular/shared/dae_dnn/results/catalog/`
- `MLPS/tabular/shared/dae_dnn/results/README.md`
- `docs/tabular_dae_dnn/README.md`
