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
- `simulation`: task-root placeholder only for W2D/STL/width_only
- `prediction`: task-root placeholder only for W2D/STL/width_only
