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

Classification is the only task with a recovered width-only prefix in the
catalog so far. That prefix is currently `d1_w1` and `d1_w2`; `w3` through
`w10` are still a TODO.

The live repeat-5 ADP run stays separate at:

- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/repeat5_v1`

The catalog folders are lightweight pointers and placeholders. The actual
archived result roots remain in the repo under their historical locations so
the live run is not disturbed.
