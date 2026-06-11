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
