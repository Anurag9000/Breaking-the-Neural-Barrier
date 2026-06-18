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

## Glaring STL TODO

The STL ablation path is intentionally still an open work item. The runner and
result layout are in place, but the full repeat-level final analysis and plots
are not yet treated as a finished deliverable.

Primary entry points:
- `run_task.py` - single task runner
- `run_all.py` - full tabular suite
- `run_stl_ablation.py` - STL ablation family generator
- `run_stl_ablation_parallel.py` - resumable STL launcher
- `run_missing_width_stl_recovery_pressure.py` - pressure-aware recovery
  launcher for the cataloged width-only and anomaly small-grid gaps
- `run_adp_w2d_suite_parallel.py` - resumable ADP width-to-depth suite
- `probe_capacity.py` - capacity probing helper
- `summarize_repeat_metrics.py` - repeat-level summary generation

The recovery launcher is GPU-first when VRAM is available, spills to CPU when
GPU admission is blocked, and keeps the host-wide gate closed only for
host-RAM or swap pressure pauses. GPU pauses only block GPU retries until a
GPU child completes cleanly.

Canonical result layout:
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`

For a fresh STL ablation, use the canonical root documented in
`../../../../docs/tabular_dae_dnn/README.md`.
