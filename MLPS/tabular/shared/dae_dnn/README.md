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
- `run_missing_width_stl_recovery_pressure.sh` - CPU-only default recovery
  launcher for the cataloged width-only and anomaly small-grid gaps
- `run_missing_width_stl_recovery_pressure_gpu_cpu.sh` - mixed GPU+CPU
  recovery launcher for the same gaps and the same result tree
- `run_stl_massive_band_01_03_fresh.sh` - strict STL band `1-3` launcher
- `run_stl_massive_band_04_06_fresh.sh` - strict STL band `4-6` launcher
- `run_stl_massive_band_07_08_fresh.sh` - strict STL band `7-8` launcher
- `run_stl_massive_band_09_10_fresh.sh` - strict STL band `9-10` launcher
- `run_adp_w2d_suite_parallel.py` - resumable ADP width-to-depth suite
- `probe_capacity.py` - capacity probing helper
- `summarize_repeat_metrics.py` - repeat-level summary generation

Windows entry points:
- `run_missing_width_stl_recovery_pressure.ps1` - native PowerShell CPU-only
  default recovery launcher
- `run_missing_width_stl_recovery_pressure_gpu_cpu.ps1` - native PowerShell
  mixed GPU+CPU recovery launcher
- `run_stl_massive_band_01_03_fresh.ps1` - native PowerShell strict STL
  band `1-3` launcher
- `run_stl_massive_band_04_06_fresh.ps1` - native PowerShell strict STL
  band `4-6` launcher
- `run_stl_massive_band_07_08_fresh.ps1` - native PowerShell strict STL
  band `7-8` launcher
- `run_stl_massive_band_09_10_fresh.ps1` - native PowerShell strict STL
  band `9-10` launcher

The `.sh` wrappers remain the canonical Linux, WSL, and Git Bash entry
points. Native `cmd.exe` and PowerShell do not execute `./.../*.sh` directly;
use the matching `.ps1` wrapper there. All wrappers call the same Python
launchers and write the same result roots.

On Windows 11, the Python runtime applies best-effort `SetPriorityClass`
process priority and `SetProcessAffinityMask` CPU affinity through the Win32
API. That is the documented equivalent surface for Linux `renice` and CPU
affinity control. Windows does not expose a direct `ionice`-style user-facing
API with the same shape; the closest documented lower-priority analogue is
`PROCESS_MODE_BACKGROUND_BEGIN`, which lowers CPU and resource scheduling
priorities, but the training launchers do not enable background mode by
default.

The CPU-only wrapper is now the default launcher for this recovery family.
It hides CUDA before bootstrap, uses the same
`MLPS/tabular/shared/dae_dnn/results/recovery/missing_width_stl_v1` root,
and preserves the same child checkpoint files as the mixed runner. The mixed
runner is a separate script that keeps CUDA visible and uses the same root,
same child directories, and same checkpoint files, so CPU and GPU runs can
resume each other without any layout fork.

Recovery wrapper contract:

1. build the recovery job list from the width-only and anomaly small-grid gaps
2. sort by existing resume state first, then by parameter count, depth, and name
3. launch children under the shared recovery root
4. write `checkpoint_last.pt`, `checkpoint_best.pt`, `candidate_state.json`,
   and `_recovery_child_process.log` under each child root
5. resume from the same model weights, optimizer state, RNG state, and child
   architecture when a checkpoint already exists
6. no LR scheduler is used in this recovery runner, so there is no scheduler
   state to restore today
7. treat GPU pressure as GPU-only admission blocking and host RAM pressure as
   a global admission block
8. ignore swap in the default wrapper path by setting swap thresholds to
   `100 / 100`
9. use `--max-active-jobs 0` as the all-visible-CPU default and keep
   `--post-launch-sample-delay-sec 60` as the standard 1 minute post-launch sample window
10. use shared platform helpers for host memory sampling and child-process
    tree termination on Linux, WSL, Git Bash, and native Windows

Canonical result layout:
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`

For a fresh STL ablation, use the canonical root documented in
`../../../../docs/tabular_dae_dnn/README.md`.
