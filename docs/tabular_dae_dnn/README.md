# Tabular DAE/DNN Guide

This is the consolidated reference for the tabular MLP experiments under
`MLPS/tabular/shared/dae_dnn/`.

## Active task set

Current STL and ADP runs should use:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`

## Canonical data mapping

- `classification` -> `Covertype`
- `autoencoding` -> `Covertype`
- `denoising` -> `Covertype`
- `anomaly` -> `Covertype`
- `simulation` -> `California Housing`
- `prediction` -> `YearPredictionMSD`

## Canonical results layout

This layout is mandatory for all MLPS tabular DAE/DNN result drops in this
repo. Keep future outputs on `main` only and use the same task-first tree.

Future runs should live under one of these roots:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`
- `MLPS/tabular/shared/dae_dnn/results/catalog/<task>/{w2d,width_only,stl_ablation}/`

The tree-level map lives in:

- [MLPS/tabular/shared/dae_dnn/results/README.md](../../MLPS/tabular/shared/dae_dnn/results/README.md)

The experiment inventory lives in:

- [docs/tabular_dae_dnn/experiment_inventory.md](experiment_inventory.md)
- [docs/tabular_dae_dnn/experiment_inventory.csv](experiment_inventory.csv)

The exact runner and export methodology lives in:

- [docs/tabular_dae_dnn/methodology_and_handoff.md](methodology_and_handoff.md)

The missing-width / small-STL recovery family uses a shared result root and
four wrappers:

- `MLPS/tabular/shared/dae_dnn/run_missing_width_stl_recovery_pressure.sh`
  is the CPU-only default launcher for Linux, WSL, and Git Bash.
- `MLPS/tabular/shared/dae_dnn/run_missing_width_stl_recovery_pressure_gpu_cpu.sh`
  is the mixed GPU+CPU launcher for Linux, WSL, and Git Bash.
- `MLPS/tabular/shared/dae_dnn/run_missing_width_stl_recovery_pressure.ps1`
  is the CPU-only default launcher for native PowerShell.
- `MLPS/tabular/shared/dae_dnn/run_missing_width_stl_recovery_pressure_gpu_cpu.ps1`
  is the mixed GPU+CPU launcher for native PowerShell.

Both wrappers point at
`MLPS/tabular/shared/dae_dnn/results/recovery/missing_width_stl_v1`, write
the same checkpoint files, and can resume each other across device changes.
The CPU-only wrapper is the recommended default when you just want the suite
to run without GPU admission.

Both wrappers now default child launchers into shared-CPU mode. The launcher
does not partition the visible CPU set across siblings unless a caller
explicitly overrides `TABULAR_CHILD_SHARED_CPU`. Each child therefore sees the
full CPU budget and the OS scheduler handles contention. The default
DataLoader worker count now also tracks the visible core count unless a caller
overrides `TABULAR_CPU_WORKERS`.

The strict STL band launchers currently cover:

- `run_stl_massive_band_01_03_fresh.sh`
- `run_stl_massive_band_04_06_fresh.sh`
- `run_stl_massive_band_07_08_fresh.sh`
- `run_stl_massive_band_09_10_fresh.sh`

On Windows 11, the runtime uses the Win32 process-priority and affinity
surface instead of Linux `renice` / `ionice`:

- `SetPriorityClass` for process priority
- `SetProcessAffinityMask` for CPU affinity

Windows does not have a direct user-facing `ionice` equivalent with the same
shape. The closest documented background-resource mode is
`PROCESS_MODE_BACKGROUND_BEGIN`, but the training wrappers keep that disabled
by default because these are throughput jobs.
The runtime also sets the current process memory priority to `normal`
through `SetProcessInformation(ProcessMemoryPriority)` so Windows keeps the
process's pages resident as long as it can.

On Linux, the runtime requests `MemorySwapMax=0` and `MemoryZSwapMax=0` when
`systemd-run --user --scope` is available. That is the repo-local
swap-control mechanism for these launchers. The repo does not try to mutate
host swap configuration or block startup based on host-global swap state.

Regenerate it with:

```bash
./.venv/bin/python scripts/update_experiment_inventory.py
```

The current recommended fresh STL root is:

```text
MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1
```

The preflight candidate-plan report lives under:

```text
MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/analysis/planned_params
```

It writes one CSV pair and one plot per task so you can inspect how the
parameter targets fall across the decades before launching the real run.

For ADP width-to-depth, use
`MLPS/tabular/shared/dae_dnn/run_adp_w2d_suite_parallel.py`.
It resumes incomplete task roots inside the current repeat, skips completed
task roots, and moves to the next repeat only after the current repeat is
fully finished.

The preferred massive STL launcher is now the pressure-aware scheduler in
`MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py`. It expands each
task/depth family into concrete STL child runs, sorts them globally
smallest-to-largest by parameter count, but it gives priority to any child
root that already has partial resume state before untouched jobs. It launches
as many children as fit, fills GPU slots first up to the configured GPU
memory threshold, and then spills additional children onto CPU while host RAM
is still below the configured resume threshold. Host memory sampling is
cross-platform through `platform_runtime.py`: Linux and WSL use `/proc`,
native Windows uses `GlobalMemoryStatusEx`, and `psutil` is used as a
fallback when available. The admission policy is split
by pressure source for device selection, but admission itself is
completion-gated: any pressure pause closes the launcher and pressure
recovery alone does not reopen it. The scheduler waits for a genuine child
completion before it allows the next launch, and that next launch resumes a
paused or partial child before it considers untouched work. If host RAM
pressure crosses the configured threshold, it pauses the largest active
child. If a child dies with a CUDA OOM signature, the scheduler pauses the
largest active GPU child, requeues the failed child at the front of the
pending queue, and retries it again only after the gate reopens on completion.
Children always resume from the same child root and reuse the normal STL
checkpoints and `ablation_state.json`. The current recovery wrapper uses a 1
minute post-launch sample window.

The launcher also writes `job_manifest.json` under each `--run-root`. That
manifest captures the fully expanded concrete job plan, so restarting the
same run root reuses the cached plan instead of recomputing the candidate
lattice. That cache is the default path now. Finished child roots remain
skipped, partial child roots resume from their saved state, and untouched
children only start when the queue reaches them in the same resume-first
order. The manifest signature is versioned with a launcher code-version token,
so a real planner change can invalidate the cache without letting harmless
runtime-only flag changes do it. Only plan-shaping inputs invalidate the
manifest: root, tasks, band, architecture grid, repeat count, and the child
training knobs that change the concrete job list. Pressure thresholds and
concurrency do not.

Runtime tuning for the tabular launchers is centralized as well:

- shell wrappers source `MLPS/tabular/shared/dae_dnn/runtime_tuning.sh`
- shell wrappers detect either `.venv/bin/python` or
  `.venv/Scripts/python.exe`, so Git Bash can use a Windows virtualenv
- native Windows should use the checked-in `.ps1` wrappers; PowerShell and
  `cmd.exe` do not execute `./.../*.sh` directly
- Python runners call `bootstrap_runtime()` before training starts; the
  top-level entrypoint re-execs itself under `systemd-run --user --scope`
  inside `app-mlps-training.slice` when the user manager supports it
- fan-out launchers set `TABULAR_CPU_JOB_CONCURRENCY` per child and assign a
  deterministic `TABULAR_CPU_AFFINITY_CPUS` slice so concurrent children
  divide the CPU thread budget and CPU placement instead of all claiming the
  machine
- concurrent launchers now allocate explicit slot indices for active children,
  so simultaneously active jobs get disjoint CPU partitions instead of hashed
  best-effort placement
- `--num-workers 0` is treated as auto-max for the tabular loaders
- the loaders use all detected logical CPU cores by default, plus persistent
  workers and prefetching when workers are enabled
- the process makes a best-effort attempt to raise its priority with `renice`
  and `ionice`, and it also enables `OMP_WAIT_POLICY=ACTIVE`,
  `OMP_PROC_BIND=spread`, and `OMP_PLACES=cores`; if the OS denies the
  change, the thread, affinity, and worker settings still apply
- the shell-side bootstrap also requests `SCHED_BATCH`, and the Python
  bootstrap attempts the same policy with `sched_setscheduler(2)`
- Linux-specific priority features are best-effort. `systemd-run`, `renice`,
  `ionice`, `SCHED_BATCH`, and CPU affinity partitioning are skipped when the
  OS does not expose those APIs. Windows still uses the same launcher
  algorithm, result tree, checkpoints, thread environment defaults, pressure
  thresholds, and process-tree termination contract.
- the repository includes
  `MLPS/tabular/shared/dae_dnn/install_linux_runtime_priority.sh` to install
  the persistent Linux-side settings used on this machine:
  `kernel.sched_autogroup_enabled=0`, `user-UID.slice` high CPU/IO weights,
  and `user@.service` delegation for `cpu cpuset io memory pids`
- `MLPS/tabular/shared/dae_dnn/smoke_runtime_priority.py` stress-tests the
  same runtime path and verifies scope placement, `SCHED_BATCH`, disjoint
  affinity slices, and high aggregate CPU utilization

Smoke, dry-run, and recovery-probe outputs are not canonical experiment
results. Keep them only long enough to validate launcher behavior, then
remove the generated scratch tree before publishing a results snapshot.

For the missing-width/small-STL recovery wrapper and the massive STL pressure
scheduler, the shared-CPU default gives every child the full visible logical
CPU set. `--max-active-jobs 0` keeps admissions open while RAM pressure
permits it. GPU admission is memory-pressure driven by default: the scheduler
keeps launching GPU children while VRAM is below the resume threshold, then
spills additional work to CPU when GPU admission is blocked. If host RAM
pressure pauses admissions, the gate stays closed until a child completes
cleanly; resumed work is always reconsidered before untouched work. Use
`--max-active-jobs <n>` or `--max-active-gpu-jobs <n>` only when you need an
explicit child-count cap.

That policy is aggressive by design. It keeps the CPU side busy when the
current workload can exploit the extra parallelism.

Key pressure-aware flags:

- `--scheduler pressure_aware`
- `--host-ram-pressure-limit-pct 85`
- `--host-ram-resume-pct 80`
- `--gpu-memory-pressure-limit-pct 85`
- `--gpu-memory-resume-pct 80`
- `--gpu-device-index 0`
- `--max-active-jobs 0` for no hard slot cap beyond RAM pressure while each
  child still sees the full visible CPU set
- `--max-retries-per-job 0` as a legacy compatibility flag; pressure-aware mode now requeues failed children indefinitely
- `--pressure-poll-interval-sec 0.5`
- `--post-launch-sample-delay-sec 30`

For a fresh full rerun of the strict `10^4..10^6` massive STL band, use:

```bash
./MLPS/tabular/shared/dae_dnn/run_stl_massive_band_04_06_fresh.sh
```

On native PowerShell, use:

```powershell
.\MLPS\tabular\shared\dae_dnn\run_stl_massive_band_04_06_fresh.ps1
```

That wrapper launches the entire strict `4-6` band from scratch into:

```text
MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06_fresh_v1
```

and keeps the current pressure-aware settings:

- tasks: all 7 tabular tasks
- parameter band: `4 6`
- repeat count: `5`
- scheduler: `pressure_aware`
- host RAM thresholds: `85 / 80`
- GPU memory thresholds: `85 / 80`
- hard active-job cap: `0` (disabled)
- batch size: `186240`

Windows portability note: native Windows can run the Python launchers and
PowerShell wrappers, but it should not be expected to match Linux throughput
bit-for-bit. PyTorch build, CUDA driver/runtime, filesystem performance,
process startup overhead, and OS scheduler policy all affect the exact rate.
The repo-side guarantee is the same algorithm, result tree, checkpoint
layout, pressure thresholds, and resume behavior.

The strict `4-6` plan generated by the current code path is `1928` child
architecture jobs. With `5` repeats per child, that is `9640` repeat phases
overall.

Each pressure-aware child also writes a local `_child_process.log` under its
child root. The scheduler uses that file to detect CUDA OOM and cuBLAS
allocation failures when deciding whether to evict a GPU peer and immediately
requeue the failed child.

Parallelism probes for the same STL bands, if you still want a fixed-slot
concurrency recommendation for a specific laptop, should live under:

```text
MLPS/tabular/shared/dae_dnn/results/stl/parallelism_probe/<band_name>
```

The probe path is now optional compatibility tooling. It still writes a
recommended fixed concurrency and still runs the heaviest candidates first
for two epochs, but the preferred real run path is the pressure-aware
scheduler rather than `--concurrency-file`.

When splitting the massive STL run across multiple laptops, stage each
parameter-decade band under its own sibling root, for example:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow01_03`
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06`
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow07_08`
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow09_10`

Merge them back into the canonical root with
`MLPS/tabular/shared/dae_dnn/merge_stl_ablation_bands.py`.

All tabular DAE/DNN loaders cap oversize batch requests to the dataset split
length, so any batch size that exceeds the available samples becomes a single
batch epoch for that split.

The planned schedule CSV for that run root is:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/planned_params_by_task_depth_width.csv`

The current canonical small-grid task roots are:

```text
MLPS/tabular/shared/dae_dnn/results/stl/small_grid/<task>
```

Use `MLPS/tabular/shared/dae_dnn/run_stl_small_grid.py` for that no-repeat
grid. The task-facing archive layout for the small grid is task-first, with
each task root containing `w2d/`, `width_only/`, and `stl_ablation/`, and
aggregate rollups living under `analysis/`.

The `simulation` and `prediction` slave-laptop follow-up has already been
assimilated into:

- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/simulation`
- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/prediction`

Its suite-level provenance remains under:

- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/analysis/simulation_prediction_v1`

The current recommended ADP root is:

```text
MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>
```

Current live ADP repeat split:

- `classification`, `autoencoding`, `generation`, `denoising`, `anomaly`: 4 repeats in the active suite
- `simulation`, `prediction`: 5 repeats in the active suite
- the first five tasks are intended to be merged later with the prior one-off W2D history so each task has 5 combined repeats in the canonical combined tree

Historical ADP width-to-depth archives are restored under:

```text
MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_anomaly_onward_gpu
MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_staged_current
```

The staged archive now uses `classification` as the visible task label.

Use the active ADP launcher configuration to determine the current repeat
split. Do not point documentation at a deleted live root.

## Big Open TODO

The massive STL ablation we planned earlier is still a glaring open item in
this repo. The schedule and resume machinery exist, but that large all-task
sweep was stopped midway and is not a finalized end-state run.
Treat it as a TODO until the full repeat-level analysis is complete.

Open items:

- verify the exact resume boundary behavior on the live STL runner
- finish the repeat-level final-loss and final-accuracy tables
- generate per-task best-repeat and worst-repeat trajectory plots
- keep all outputs under the canonical `results/stl/ablation/` layout

Historical STL outputs from the previous top-level run root were moved to:

```text
MLPS/tabular/shared/dae_dnn/results/archive/stl_ablation_parameter_matched_gpu_serial
```

The legacy sweep is restored under the canonical `classification` archive root:

```text
MLPS/tabular/shared/dae_dnn/results/archive/classification_trial1
```

The repo-visible root is `classification_trial1`.

That lightweight historical STL archive is not the massive all-task sweep. It
contains the older small study for:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`

It does not contain dedicated STL ablation roots for `simulation` or
`prediction`.

`simulation` and `prediction` are now part of the recovered small-grid STL
archive. There is still no archived one-off W2D root for them in the current
repo.

The curated result catalog now lives under:

```text
MLPS/tabular/shared/dae_dnn/results/catalog/
```

That catalog exposes the current task groups:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`

The live ADP W2D suite uses the current launcher-selected `<suite_name>` under
`results/adp/w2d/`. Do not hardcode a deleted run root here.

Result structure rule:

- top level: task name
- inside each task: `w2d/`, `width_only/`, `stl_ablation/`
- `width_only/` and `stl_ablation/`: depth folders first, then width folders
- rollups and cross-task summaries: `analysis/`
- no parallel branch-based result trees for the same MLPS family
- commit and publish the final result state on `main`

Recovered small-grid coverage today:

- `classification`: task-first archive present; STL ablation and W2D recovered; width-only is a placeholder
- `autoencoding`: task-first archive present; STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `generation`: task-first archive present; STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `denoising`: task-first archive present; STL ablation, W2D, and width-only recovered for depths `d1` through `d10`
- `anomaly`: task-first archive present; STL ablation and W2D recovered; width-only is a placeholder
- `simulation`: task-first archive present; STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only are placeholders
- `prediction`: task-first archive present; STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only are placeholders

## Resume rule

- same `--run-root` means resume
- new `--run-root` means fresh run
- do not change task order or repeat count while resuming
- keep `main` as the only remote branch

## Notes

- Final STL/ADP summaries should use repeat-level final metrics, not
  epoch-to-epoch variance.
- The intended future report for each `(task, depth, width)` family is:
  - mean final loss
  - variance across repeats
  - loss spread between best and worst repeats
  - same for accuracy when applicable
- Plotting should stay task-wise and reuse the same architecture color across
  best-repeat and worst-repeat trajectories.

## Legacy doc pointers

The old per-folder handoff notes were collapsed into this guide and the
results-tree README:

- `MLPS/tabular/shared/dae_dnn/DEFAULT_TASKS.md`
- `MLPS/tabular/shared/dae_dnn/EXPERIMENT_HANDOFF.md`
- `MLPS/tabular/shared/dae_dnn/RUN_SPLIT_PLAN.md`
- `MLPS/tabular/shared/dae_dnn/not_accomplished.md`
