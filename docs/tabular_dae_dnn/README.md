# Tabular DAE/DNN Guide

This is the consolidated reference for the tabular MLP experiments under
`MLPS/tabular/shared/dae_dnn/`.

## Active task set

Current STL and ADP runs should use:

- `classification`
- `autoencoding`
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

Future runs should live under one of these roots:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`

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

The current recommended small-grid follow-up root is:

```text
MLPS/tabular/shared/dae_dnn/results/stl/small_grid/simulation_prediction_v1
```

Use `MLPS/tabular/shared/dae_dnn/run_stl_small_grid.py` for that no-repeat
grid.

The current recommended ADP root is:

```text
MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>
```

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

`simulation` and `prediction` are not part of the recovered small STL archive
and there is no archived one-off W2D root for them in the current repo.
If you need to run them, use the separate no-repeat small-grid runner
documented in `docs/tabular_dae_dnn/methodology_and_handoff.md`.

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
