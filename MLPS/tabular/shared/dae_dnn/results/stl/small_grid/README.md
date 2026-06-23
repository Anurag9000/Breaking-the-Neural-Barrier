# Small STL Width Sweep Archive

This archive is organized task-first. Each task root contains:
- `w2d/`
- `width_only/`
- `stl_ablation/`

The recovered historical data lives under the task root for the relevant mode.
Cross-task rollups live under `analysis/`.

Layout rules:

- `w2d/` contains the archived ADP width-to-depth phase trees for the task.
- `width_only/` is organized by depth folders, then width folders.
- `stl_ablation/` is organized by depth folders, then width folders.

The old flat `csv/` and `graphs/` layout has been folded into this structure.

Recovered task coverage today:

- `classification`: STL ablation and W2D recovered; width-only is a placeholder
- `autoencoding`: STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `generation`: STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `denoising`: STL ablation, W2D, and width-only recovered for depths `d1` through `d10`
- `anomaly`: STL ablation and W2D recovered; width-only is a placeholder
- `simulation`: STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only remain placeholders
- `prediction`: STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only remain placeholders

The `simulation` and `prediction` slave-laptop follow-up was assimilated into
the canonical task roots:

- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/simulation`
- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/prediction`

Suite-level provenance for that follow-up is preserved under:

- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/analysis/simulation_prediction_v1`
