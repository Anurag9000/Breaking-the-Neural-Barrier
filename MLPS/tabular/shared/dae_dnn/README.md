# Tabular DAE/DNN

This folder contains the tabular MLP runners, ADP search code, and experiment
helpers for the non-vision benchmark suite.

Canonical guide:
- [docs/tabular_dae_dnn/README.md](../../../../docs/tabular_dae_dnn/README.md)

Current active STL and ADP task set:
- classification
- autoencoding
- denoising
- anomaly
- simulation
- prediction

Generation remains archived in historical result trees and is not part of the
current STL sweep.

Primary entry points:
- `run_task.py` - single task runner
- `run_all.py` - full tabular suite
- `run_stl_ablation.py` - STL ablation family generator
- `run_stl_ablation_parallel.py` - resumable STL launcher
- `run_adp_w2d_suite_parallel.py` - resumable ADP width-to-depth suite
- `probe_capacity.py` - capacity probing helper
- `summarize_repeat_metrics.py` - repeat-level summary generation

Canonical result layout:
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`

For a fresh STL ablation, use the canonical root documented in
`../../../../docs/tabular_dae_dnn/README.md`.
