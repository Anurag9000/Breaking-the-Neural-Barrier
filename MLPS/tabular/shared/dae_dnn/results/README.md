# Results Layout

The repo now has two distinct views of results:

- a curated catalog under `MLPS/tabular/shared/dae_dnn/results/catalog/`
- the current resumable run tree under `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`

The catalog is the repo-facing organization layer for historical and supporting
results. Keep any active run tree separate from the catalog while it is still
in progress.

The canonical result layout for current and future tabular DAE/DNN runs is:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`

Use the active suite roots for resumable runs. Put old runs under `archive/`
instead of adding more sibling result trees at the top level.

Current recommended fresh roots:

- STL: `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1`
- small-grid STL: `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/simulation_prediction_v1`
- ADP: `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`

The ADP W2D suite uses task-specific repeat counts. Keep the current launcher
configuration in sync with the active run root and do not reuse a deleted root.

Historical ADP W2D archives restored into the repo:

- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_anomaly_onward_gpu`
- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_staged_current`

The staged legacy archive now exposes `classification` at the repo-visible root.

Catalog view:

- `MLPS/tabular/shared/dae_dnn/results/catalog/classification`
- `MLPS/tabular/shared/dae_dnn/results/catalog/autoencoding`
- `MLPS/tabular/shared/dae_dnn/results/catalog/generation`
- `MLPS/tabular/shared/dae_dnn/results/catalog/denoising`
- `MLPS/tabular/shared/dae_dnn/results/catalog/anomaly`
- `MLPS/tabular/shared/dae_dnn/results/catalog/simulation`
- `MLPS/tabular/shared/dae_dnn/results/catalog/prediction`

`simulation` and `prediction` are not part of the recovered small STL archive.
They should be treated as absent from the historical STL material until new
results are generated.

For the missing `simulation` and `prediction` follow-up, use the dedicated
no-repeat small-grid runner. It writes to `results/stl/small_grid/<suite_name>/`
and nests each candidate under a depth/width path.

STL ablation is still a flagged TODO in the repo docs. Do not treat the
archived STL tree as finished state; it is the history backing the current
fresh run layout and the remaining analysis work.

The planned STL schedule lives at:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/planned_params_by_task_depth_width.csv`

Archived outputs already moved there include:

- `MLPS/tabular/shared/dae_dnn/results/archive/stl_ablation_parameter_matched_gpu_serial`
- `MLPS/tabular/shared/dae_dnn/results/archive/classification_trial1`

`classification_trial1` is the older lightweight STL study, not the massive
all-task ablation. It covers:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`

It does not include dedicated STL roots for `simulation` or `prediction`.

Experiment catalog files:

- `docs/tabular_dae_dnn/experiment_inventory.md`
- `docs/tabular_dae_dnn/experiment_inventory.csv`

Regenerate the catalog with:

```bash
./.venv/bin/python scripts/update_experiment_inventory.py
```
