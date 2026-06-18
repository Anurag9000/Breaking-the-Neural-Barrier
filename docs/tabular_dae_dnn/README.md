# Tabular DAE/DNN Guide

This is the consolidated reference for the tabular MLP experiments under
`MLPS/tabular/shared/dae_dnn/`.

## Active task set

Current STL and ADP runs should use:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`

`generation` is archived in older runs and remains available in the historical
goliath trees, but it is not part of the current STL sweep.

## Canonical data mapping

- `classification` -> `Covertype`
- `autoencoding` -> `Covertype`
- `denoising` -> `Covertype`
- `anomaly` -> `Covertype`
- `simulation` -> `California Housing`
- `prediction` -> `YearPredictionMSD`

## Canonical results layout

This layout is mandatory for all MLPS tabular DAE/DNN result drops in this
repo. Keep future outputs on `main` only and use the same task-first tree.

Future runs should live under one of these roots:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`
- `MLPS/tabular/shared/dae_dnn/results/catalog/<task>/{w2d,width_only,stl_ablation}/`

The tree-level map lives in:

- [MLPS/tabular/shared/dae_dnn/results/README.md](../../MLPS/tabular/shared/dae_dnn/results/README.md)

The experiment inventory lives in:

- [docs/tabular_dae_dnn/experiment_inventory.md](experiment_inventory.md)
- [docs/tabular_dae_dnn/experiment_inventory.csv](experiment_inventory.csv)

The exact runner and export methodology lives in:

- [docs/tabular_dae_dnn/methodology_and_handoff.md](methodology_and_handoff.md)

Regenerate it with:

```bash
./.venv/bin/python scripts/update_experiment_inventory.py
```

The current recommended fresh STL root is:

```text
MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1
```

The preflight candidate-plan report lives under:

```text
MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/analysis/planned_params
```

It writes one CSV pair and one plot per task so you can inspect how the
parameter targets fall across the decades before launching the real run.

For ADP width-to-depth, use
`MLPS/tabular/shared/dae_dnn/run_adp_w2d_suite_parallel.py`.
It resumes incomplete task roots inside the current repeat, skips completed
task roots, and moves to the next repeat only after the current repeat is
fully finished.

The preferred massive STL launcher is now the pressure-aware scheduler in
`MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py`. It expands each
task/depth family into concrete STL child runs, sorts them globally
smallest-to-largest by parameter count, but it gives priority to any child
root that already has partial resume state before untouched jobs. It launches
as many children as fit, fills GPU slots first up to the configured GPU
memory threshold, and then spills additional children onto CPU while host RAM
is still below the configured resume threshold. After the initial fill, the
admission rule is completion-gated: any pressure pause or retryable child
failure closes the admission window immediately. The scheduler only reopens
the window after a genuine child completion, at which point it resumes paused
or partial children before it considers untouched jobs. If host RAM pressure
crosses the configured threshold, it pauses the largest active child. If a
child dies with a CUDA OOM signature, the scheduler pauses the largest active
GPU child, requeues the failed child at the front of the pending queue, and
retries it again as soon as a device slot becomes available. Children always
resume from the same child root and reuse the normal STL checkpoints and
`ablation_state.json`. For the slower laptop split, set
`--pressure-settle-sec 120` so each launch gets a two-minute pressure settle
window.

Key pressure-aware flags:

- `--scheduler pressure_aware`
- `--host-ram-pressure-limit-pct 90`
- `--host-ram-resume-pct 85`
- `--gpu-memory-pressure-limit-pct 90`
- `--gpu-memory-resume-pct 85`
- `--gpu-device-index 0`
- `--max-active-jobs 0` for no hard slot cap beyond RAM pressure
- `--max-retries-per-job 0` as a legacy compatibility flag; pressure-aware mode now requeues failed children indefinitely
- `--pressure-poll-interval-sec 0.5`
- `--pressure-settle-sec 1.0`

Each pressure-aware child also writes a local `_child_process.log` under its
child root. The scheduler uses that file to detect CUDA OOM and cuBLAS
allocation failures when deciding whether to evict a GPU peer and immediately
requeue the failed child.

Parallelism probes for the same STL bands, if you still want a fixed-slot
concurrency recommendation for a specific laptop, should live under:

```text
MLPS/tabular/shared/dae_dnn/results/stl/parallelism_probe/<band_name>
```

The probe path is now optional compatibility tooling. It still writes a
recommended fixed concurrency and still runs the heaviest candidates first
for two epochs, but the preferred real run path is the pressure-aware
scheduler rather than `--concurrency-file`.

When splitting the massive STL run across multiple laptops, stage each
parameter-decade band under its own sibling root, for example:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow01_03`
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06`
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow07_08`
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow09_10`

Merge them back into the canonical root with
`MLPS/tabular/shared/dae_dnn/merge_stl_ablation_bands.py`.

The planned schedule CSV for that run root is:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/planned_params_by_task_depth_width.csv`

The current canonical small-grid task roots are:

```text
MLPS/tabular/shared/dae_dnn/results/stl/small_grid/<task>
```

Use `MLPS/tabular/shared/dae_dnn/run_stl_small_grid.py` for that no-repeat
grid. The task-facing archive layout for the small grid is task-first, with
each task root containing `w2d/`, `width_only/`, and `stl_ablation/`, and
aggregate rollups living under `analysis/`.

The `simulation` and `prediction` slave-laptop follow-up has already been
assimilated into:

- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/simulation`
- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/prediction`

Its suite-level provenance remains under:

- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/analysis/simulation_prediction_v1`

The current recommended ADP root is:

```text
MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>
```

Current live ADP repeat split:

- `classification`, `autoencoding`, `generation`, `denoising`, `anomaly`: 4 repeats in the active suite
- `simulation`, `prediction`: 5 repeats in the active suite
- the first five tasks are intended to be merged later with the prior one-off W2D history so each task has 5 combined repeats in the canonical combined tree

Historical ADP width-to-depth archives are restored under:

```text
MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_anomaly_onward_gpu
MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_staged_current
```

The staged archive now uses `classification` as the visible task label.

Use the active ADP launcher configuration to determine the current repeat
split. Do not point documentation at a deleted live root.

## Big Open TODO

The massive STL ablation we planned earlier is still a glaring open item in
this repo. The schedule and resume machinery exist, but that large all-task
sweep was stopped midway and is not a finalized end-state run.
Treat it as a TODO until the full repeat-level analysis is complete.

Open items:

- verify the exact resume boundary behavior on the live STL runner
- finish the repeat-level final-loss and final-accuracy tables
- generate per-task best-repeat and worst-repeat trajectory plots
- keep all outputs under the canonical `results/stl/ablation/` layout

Historical STL outputs from the previous top-level run root were moved to:

```text
MLPS/tabular/shared/dae_dnn/results/archive/stl_ablation_parameter_matched_gpu_serial
```

The legacy sweep is restored under the canonical `classification` archive root:

```text
MLPS/tabular/shared/dae_dnn/results/archive/classification_trial1
```

The repo-visible root is `classification_trial1`.

That lightweight historical STL archive is not the massive all-task sweep. It
contains the older small study for:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`

It does not contain dedicated STL ablation roots for `simulation` or
`prediction`.

`simulation` and `prediction` are now part of the recovered small-grid STL
archive. There is still no archived one-off W2D root for them in the current
repo.

The curated result catalog now lives under:

```text
MLPS/tabular/shared/dae_dnn/results/catalog/
```

That catalog exposes the current task groups:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`

The live ADP W2D suite uses the current launcher-selected `<suite_name>` under
`results/adp/w2d/`. Do not hardcode a deleted run root here.

Result structure rule:

- top level: task name
- inside each task: `w2d/`, `width_only/`, `stl_ablation/`
- `width_only/` and `stl_ablation/`: depth folders first, then width folders
- rollups and cross-task summaries: `analysis/`
- no parallel branch-based result trees for the same MLPS family
- commit and publish the final result state on `main`

Recovered small-grid coverage today:

- `classification`: task-first archive present; STL ablation and W2D recovered; width-only is a placeholder
- `autoencoding`: task-first archive present; STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `generation`: task-first archive present; STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `denoising`: task-first archive present; STL ablation, W2D, and width-only recovered for depths `d1` through `d10`
- `anomaly`: task-first archive present; STL ablation and W2D recovered; width-only is a placeholder
- `simulation`: task-first archive present; STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only are placeholders
- `prediction`: task-first archive present; STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only are placeholders

## Resume rule

- same `--run-root` means resume
- new `--run-root` means fresh run
- do not change task order or repeat count while resuming
- keep `main` as the only remote branch

## Notes

- Final STL/ADP summaries should use repeat-level final metrics, not
  epoch-to-epoch variance.
- The intended future report for each `(task, depth, width)` family is:
  - mean final loss
  - variance across repeats
  - loss spread between best and worst repeats
  - same for accuracy when applicable
- Plotting should stay task-wise and reuse the same architecture color across
  best-repeat and worst-repeat trajectories.

## Legacy doc pointers

The old per-folder handoff notes were collapsed into this guide and the
results-tree README:

- `MLPS/tabular/shared/dae_dnn/DEFAULT_TASKS.md`
- `MLPS/tabular/shared/dae_dnn/EXPERIMENT_HANDOFF.md`
- `MLPS/tabular/shared/dae_dnn/RUN_SPLIT_PLAN.md`
- `MLPS/tabular/shared/dae_dnn/not_accomplished.md`
