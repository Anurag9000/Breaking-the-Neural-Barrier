# Curated Results Catalog

This tree is the repo-visible organization layer for historical and supporting
tabular DAE/DNN results.

Layout:

- Each task folder has:
  - `w2d/`
  - `width_only/`
  - `stl_ablation/`

Storage conventions:

- `w2d/combined_5repeat/` is the normalized five-repeat view for a task.
- Repeat directories use stable names such as `repeat_01` through `repeat_05`.
- Provenance-preserving aliases may appear, for example `repeat_05_legacy`,
  `repeat_05_legacy_representation`, or `repeat_05_operator_declared`.
- `width_only/by_depth_d01_d06/` groups depth sweeps by `d01` through `d06`.
- `stl_ablation/small_grid/` points at the recovered small STL grid sources.
- `stl_ablation/massive_parammatched_decade_v1/` points at the merged massive
  band catalog.

The catalog folders are lightweight pointers and placeholders. The actual
archived result roots remain in the repo under their historical locations so
the live run is not disturbed.

For the current forensic coverage state, including which payloads were found
under legacy names and which expected runs are still missing, read:

- `COVERAGE_REPORT.md`
