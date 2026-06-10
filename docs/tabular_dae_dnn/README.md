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

The planned schedule CSV for that run root is:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/planned_params_by_task_depth_width.csv`

The current recommended ADP root is:

```text
MLPS/tabular/shared/dae_dnn/results/adp/w2d/repeat5_v1
```

Historical ADP width-to-depth archives are restored under:

```text
MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_anomaly_onward_gpu
MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_staged_current
```

The staged archive now uses `classification` as the visible task label.

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

- `classification` / legacy `representation`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`

It does not contain dedicated STL ablation roots for `simulation` or
`prediction`.

The recovered classification width-only lineage is partial. The repo-visible
archive currently contains the `d1_w1` and `d1_w2` prefix under
`results/archive/goliath_w2d_staged_current/classification/ae_width_only`.
Widths `w3` through `w10` remain a TODO.

If you want the separate small STL follow-up run for the missing
`simulation` and `prediction` tasks, use the exact command in
`docs/tabular_dae_dnn/methodology_and_handoff.md`. That follow-up is distinct
from the massive STL ablation TODO.

The curated result catalog now lives under:

```text
MLPS/tabular/shared/dae_dnn/results/catalog/
```

That catalog exposes the current task groups:

- `representation`
- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`

The live repeat-5 ADP W2D suite remains separate at:

```text
MLPS/tabular/shared/dae_dnn/results/adp/w2d/repeat5_v1
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
