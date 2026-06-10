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
- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`

The tree-level map lives in:

- [MLPS/tabular/shared/dae_dnn/results/README.md](../../MLPS/tabular/shared/dae_dnn/results/README.md)

The experiment inventory lives in:

- [docs/tabular_dae_dnn/experiment_inventory.md](experiment_inventory.md)
- [docs/tabular_dae_dnn/experiment_inventory.csv](experiment_inventory.csv)

Regenerate it with:

```bash
./.venv/bin/python scripts/update_experiment_inventory.py
```

The current recommended fresh STL root is:

```text
MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1
```

The planned schedule CSV for that run root is:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/planned_params_by_task_depth_width.csv`

The current recommended ADP root is:

```text
MLPS/tabular/shared/dae_dnn/results/adp/w2d/repeat5_v1
```

## Big Open TODO

STL ablation is still a glaring open item in this repo. The schedule and
resume machinery exist, but the suite is not yet a finalized end-state run.
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

The legacy `representation` sweep is restored under:

```text
MLPS/tabular/shared/dae_dnn/results/archive/representation_trial1
```

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
