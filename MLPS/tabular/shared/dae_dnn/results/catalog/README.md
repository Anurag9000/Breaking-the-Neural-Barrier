# Curated Results Catalog

This tree is the repo-visible organization layer for historical and supporting
tabular DAE/DNN results.

Layout:

- `representation/` is intentionally empty. The legacy `representation` label
  has been normalized to `classification`.
- Each task folder has:
  - `adpw2d/`
  - `adp_width_only/`
  - `stl_ablation/`

The live repeat-5 ADP run stays separate at:

- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/repeat5_v1`

The catalog folders are lightweight pointers and placeholders. The actual
archived result roots remain in the repo under their historical locations so
the live run is not disturbed.

