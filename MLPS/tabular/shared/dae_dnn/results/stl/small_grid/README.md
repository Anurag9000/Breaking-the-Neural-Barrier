# Small STL Grid

This tree is the canonical home for the lightweight no-repeat STL follow-up.
It is separate from the massive repeat-based STL ablation.

Recommended layout:

- `results/stl/small_grid/<suite_name>/<task>/stl_ablation/d03/w064/cand_000/`
- one candidate per `(task, depth, width)`
- no repeat directories

Archived historical reference:

- `MLPS/tabular/shared/dae_dnn/results/archive/classification_trial1`

Recommended grid:

- depths: `3`, `4`, `6`, `8`, `10`
- widths: `64`, `96`, `128`, `160`, `192`, `224`, `256`

Use `MLPS/tabular/shared/dae_dnn/run_stl_small_grid.py` to generate new runs
in this layout.
