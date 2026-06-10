# Results Layout

The canonical result layout for current and future tabular DAE/DNN runs is:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`

Use the active suite roots for resumable runs. Put old runs under `archive/`
instead of adding more sibling result trees at the top level.

Current recommended fresh roots:

- STL: `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1`
- ADP: `MLPS/tabular/shared/dae_dnn/results/adp/w2d/repeat5_v1`

Historical ADP W2D archives restored into the repo:

- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_anomaly_onward_gpu`
- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_staged_current`

The staged legacy archive now exposes `classification` instead of the old
`representation` task label at the repo-visible root.

STL ablation is still a flagged TODO in the repo docs. Do not treat the
archived STL tree as finished state; it is the history backing the current
fresh run layout and the remaining analysis work.

The planned STL schedule lives at:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/planned_params_by_task_depth_width.csv`

Archived outputs already moved there include:

- `MLPS/tabular/shared/dae_dnn/results/archive/stl_ablation_parameter_matched_gpu_serial`
- `MLPS/tabular/shared/dae_dnn/results/archive/classification_trial1`

Experiment catalog files:

- `docs/tabular_dae_dnn/experiment_inventory.md`
- `docs/tabular_dae_dnn/experiment_inventory.csv`

Regenerate the catalog with:

```bash
./.venv/bin/python scripts/update_experiment_inventory.py
```
