# Tabular DAE/DNN Methodology and Handoff

This document records the exact code paths and workflow used for the tabular
MLP experiments in this repo. Use it as the reference when running the same
study on another machine.

The results layout described here is the only accepted structure for future
MLPS tabular DAE/DNN outputs in this repo, and the final state should be
published on `main` only.

## Canonical task names

Current task names:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`

## Exact code paths to use

Task definitions and dataset mapping:

- `MLPS/tabular/shared/dae_dnn/tasks.py`

STL sweep orchestration:

- `MLPS/tabular/shared/dae_dnn/run_stl_ablation.py`
- `MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py`
- `MLPS/tabular/shared/dae_dnn/run_stl_parallelism_probe.py`
- `MLPS/tabular/shared/dae_dnn/generate_stl_planned_params_report.py`
- `MLPS/tabular/shared/dae_dnn/run_stl_small_grid.py`

ADP width-to-depth orchestration:

- `MLPS/tabular/shared/dae_dnn/run_task.py`
- `MLPS/tabular/shared/dae_dnn/adp_search.py`
- `utils/adp_contract.py`
- `MLPS/tabular/shared/dae_dnn/run_adp_w2d_suite_parallel.py`

Historical ADP W2D archives already restored into the repo:

- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_anomaly_onward_gpu`
- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_staged_current`

The staged legacy lineage is exposed in the repo-visible archive root as
`classification` for that task lineage.

## ADP Resume Order

Use `run_adp_w2d_suite_parallel.py` for resumable ADP width-to-depth suites.
It works repeat by repeat:

- inside a repeat, completed task roots are skipped
- incomplete task roots are resumed from their saved task root
- new task roots in that same repeat are started up to `--concurrency`
- the launcher does not open the next repeat until the current repeat is
  fully finished

For the current `repeat4_plus_sim_pred5_v1` suite, the saved snapshot has:

- `repeat_01/denoising` incomplete
- all other `repeat_01` tasks complete

So the next resume starts with `repeat_01/denoising` only. After that repeat
finishes, the launcher opens `repeat_02` and starts all seven tasks there.

## Width-to-depth rule

For width-to-depth runs, a depth expansion is not treated as a terminal step.
After the new layer is added, the controller continues widening neuron by
neuron until the model is uniform again, and only then does the patience
counter resume. The width-stage failure and margin counters are reset after a
completed depth/fill cycle, so a model like `[29, 29, 29]` can continue to
`[30, 29, 29]`, `[30, 30, 29]`, and `[30, 30, 30]` before width patience is
allowed to stop the search. That prevents the run from quitting while the
post-depth warmup is still in progress.

## Small historical STL archive

This is separate from the massive STL ablation TODO.

The lightweight historical STL archive at `results/archive/classification_trial1`
is a one-off fixed-grid summary, not a repeat suite. It covers the following
tasks:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`

The archived STL ablation grid is sparse and fixed:

- depths: `3`, `4`, `6`, `8`, `10`
- widths: `64`, `96`, `128`, `160`, `192`, `224`, `256`
- repeats: none

The repo-facing task layout for this archive is task-first. Each task root is
organized as:

- `w2d/`
- `width_only/`
- `stl_ablation/`

Within `stl_ablation/`, depths are the outer folders and widths are the
nested subfolders. Within `width_only/`, depth folders are the outer layer
and the width candidates are nested below that.

That exact nesting is the required structure for any future MLPS tabular
result drop:

- task at the top
- `w2d/`, `width_only/`, `stl_ablation/` immediately below
- depth folders inside `width_only/` and `stl_ablation/`
- width folders inside each depth folder
- cross-task rollups under `analysis/`
- publish the final merged tree on `main`

Recovered small-grid coverage today:

- `classification`: STL ablation and W2D recovered; width-only is a placeholder
- `autoencoding`: STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `generation`: STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `denoising`: STL ablation, W2D, and width-only recovered for depths `d1` through `d10`
- `anomaly`: STL ablation and W2D recovered; width-only is a placeholder
- `simulation`: STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only are placeholders
- `prediction`: STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only are placeholders

The `simulation` and `prediction` slave-laptop follow-up has already been
assimilated into the canonical task roots:

- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/simulation`
- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/prediction`

Its suite-level provenance remains under:

- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/analysis/simulation_prediction_v1`

Same `--run-root` means resume. Keep this separate from the massive all-task
STL sweep.

## Small STL follow-up on slave laptops

If you want to rerun the `simulation` and `prediction` small-grid follow-up on
a slave machine, use the dedicated no-repeat grid runner and keep it out of
the massive all-task STL ablation tree.

Use this command:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_small_grid.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/small_grid/simulation_prediction_v1 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/archive/classification_trial1 \
  --tasks simulation prediction \
  --depths 3 4 6 8 10 \
  --widths 64 96 128 160 192 224 256 \
  --num-workers 0 \
  --patience 10 \
  --batch-size 186240 \
  --max-epochs 100000000
```

This follow-up is separate from the massive all-task STL TODO and should not
be mixed with the live repeat-4 ADP run. It is one run per `(task, depth,
width)` candidate, with no repeats.

## Massive STL parameter-band split

The massive STL ablation is still the main long-running study, but it is now
split by parameter-count decade bands so multiple laptops can work on
disjoint slices in parallel and the results can be merged later.

Use the same runner, but cap the parameter-decade range per machine:

- machine 1: parameter decades `1` through `3`
- machine 2: parameter decades `4` through `6`
- machine 3: parameter decades `7` and `8`
- machine 4: parameter decades `9` and `10`

The launcher now accepts `--param-band START END`, where the values are
parameter-count exponents. For example, `--param-band 1 3` keeps targets in
the `10^1` through `10^3` range.

For the real massive STL ablation, training is intended to stop by early
stopping only. There is no intended short epoch cap on the actual ablation
run. The real runner default is effectively unbounded (`--max-epochs
100000000`) so patience is the stopping mechanism. The only intentionally
short run in this workflow is the parallelism probe, which is capped at two
epochs on purpose.

Before starting the real ablation on a given laptop, prefer the
pressure-aware scheduler in
`MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py`. It now expands
the concrete STL child runs for the chosen band, sorts them globally from
smallest to largest parameter count, but partial child roots with existing
resume state are always considered before untouched jobs. It opportunistically
fills the machine with as many children as fit, using GPU first until the GPU
memory resume threshold is reached and then spilling additional children onto
CPU while host RAM is still below its resume threshold. It monitors host RAM
through `platform_runtime.py`: `/proc/meminfo` on Linux/WSL,
`GlobalMemoryStatusEx` on native Windows, and `psutil` as a fallback. GPU
memory is sampled through `nvidia-smi`. After the initial saturation phase,
admission is pressure-gated: a host-RAM pause closes the admission window for
new work until host pressure falls back under the resume threshold. GPU
pressure only blocks GPU admissions; CPU spillover can continue whenever host
pressure is healthy. The launcher does not use the pause itself as a launch
signal. That next launch attempt resumes a paused or partial child before it
considers untouched work. If used RAM crosses the configured host threshold,
it requests a pause on the largest active child, terminates only that child
process group, and requeues that same child root. If a child exits with a CUDA
OOM or cuBLAS allocation failure, the scheduler requests a pause on the
largest active GPU child, requeues the failed child at the front of the
pending queue, and retries it as soon as resources allow. The launcher also
persists its expanded job plan as
`job_manifest.json` under the selected `--run-root`. On restart it reloads
that manifest instead of recomputing the candidate lattice, so the same
`--run-root` resumes the exact same concrete job set. This is the default
planner path now; the launcher plans once, writes the manifest, and then
reuses that manifest on later boots. Finished child roots stay skipped,
resumable roots keep their existing checkpoints and `ablation_state.json`,
and untouched jobs remain untouched until the launcher gets to them in the
same resume-first, low-parameter-first order. Only plan-shaping inputs
invalidate the manifest: root, tasks, band, architecture grid, repeat count,
data/result roots, and child-training knobs that change the concrete job list.
The signature is also gated by a launcher code-version token, so a real plan
change can force a rebuild while runtime-only flag changes still reuse the
cached plan. Scheduler-only knobs such as concurrency and pressure thresholds
do not.
The checkpoint boundary is the last completed batch or epoch that reached
`checkpoint_last.pt`; an OOM that kills the process before the next save
resumes from that last durable state, not from the exact Python instruction
that faulted. On the slower laptop split, set
`--pressure-settle-sec 30` to give each launch a shorter settle window.

Runtime policy for the tabular runners is centralized:

- shell launchers source `MLPS/tabular/shared/dae_dnn/runtime_tuning.sh`
- shell launchers detect either `.venv/bin/python` or
  `.venv/Scripts/python.exe`
- native Windows should use the checked-in `.ps1` wrappers; PowerShell and
  `cmd.exe` do not execute `./.../*.sh` directly
- Python runners call `bootstrap_runtime()` before they build tasks or start
  the main loop; the top-level process re-execs itself under
  `systemd-run --user --scope` in `app-mlps-training.slice` when the user
  manager supports the required resource controls
- fan-out launchers set `TABULAR_CPU_JOB_CONCURRENCY` for each child and
  assign a deterministic `TABULAR_CPU_AFFINITY_CPUS` slice so concurrent
  children divide the CPU thread budget and CPU placement instead of all
  claiming the machine
- concurrent launchers allocate explicit slot indices for active children so
  simultaneously active jobs get disjoint CPU partitions rather than hashed
  best-effort placement
- `--num-workers 0` resolves to the full logical CPU count unless an explicit
  positive worker count is passed
- the loaders use persistent workers and prefetching when worker processes are
  enabled
- the process attempts best-effort `renice -20` and `ionice -c2 -n0`, and it
  also enables `OMP_WAIT_POLICY=ACTIVE`, `OMP_PROC_BIND=spread`, and
  `OMP_PLACES=cores`; the run still proceeds if the OS denies those calls
- both shell and Python bootstraps attempt `SCHED_BATCH` for long-running
  throughput-oriented CPU work where the OS exposes it
- Windows does not provide the Linux `systemd-run`, `renice`, `ionice`,
  `SCHED_BATCH`, `/proc`, or `sched_setaffinity` behavior used on this
  laptop. Those pieces are best-effort and skipped there; the cross-platform
  contract is the same launcher algorithm, same run roots, same checkpoints,
  same thread environment defaults, same pressure thresholds, and safe
  Windows process-tree termination.
- `MLPS/tabular/shared/dae_dnn/install_linux_runtime_priority.sh` installs the
  persistent host settings used here: `kernel.sched_autogroup_enabled=0`,
  `user-UID.slice` high CPU/IO weights plus `TasksMax=infinity`, and
  `user@.service` delegation of `cpu cpuset io memory pids`
- `MLPS/tabular/shared/dae_dnn/smoke_runtime_priority.py` verifies the runtime
  path end to end: scoped execution, `SCHED_BATCH`, disjoint affinity slices,
  and high aggregate CPU utilization under synthetic load

Smoke, dry-run, and recovery-probe outputs are transient validation artifacts.
They are not part of the canonical experiment record and should be deleted
before a results tree is treated as final.

## Missing-width recovery wrapper

The missing-width / small-STL recovery family is split into two launchers that
share one result tree and one checkpoint contract:

- `MLPS/tabular/shared/dae_dnn/run_missing_width_stl_recovery_pressure.sh`
  is the CPU-only default.
- `MLPS/tabular/shared/dae_dnn/run_missing_width_stl_recovery_pressure_gpu_cpu.sh`
  is the mixed GPU+CPU runner.

Both wrappers write into
`MLPS/tabular/shared/dae_dnn/results/recovery/missing_width_stl_v1/` and both
resume from the same child roots. A run can move from CPU to GPU or GPU to CPU
without changing the directory layout or the checkpoint files.

The recovery algorithm is:

1. build the width-only and small-STL job list
2. sort by existing resume state first, then by parameter count, depth, and name
3. launch children under the shared recovery root
4. write `checkpoint_last.pt`, `checkpoint_best.pt`, `candidate_state.json`,
   and `_recovery_child_process.log` under each child root
5. load the same model weights, optimizer state, and RNG state when resuming
6. keep the architecture fixed to the candidate root being resumed
7. no LR scheduler is used in this recovery runner, so there is no scheduler
   state to restore today
8. treat host-RAM pressure as a global admission gate
9. treat GPU memory pressure as a GPU-only admission gate in the mixed runner
10. ignore swap by default in the wrapper path by setting swap thresholds to
   `100 / 100`

The default CPU-only wrapper hides CUDA before bootstrap and therefore never
sees a GPU. The mixed runner leaves CUDA visible and uses the same root so a
GPU run can be resumed on CPU later, and vice versa.

The key knobs are:

- `--max-active-jobs 0` for all visible logical CPUs as job lanes
- `--max-active-gpu-jobs 0` for memory-driven GPU concurrency in the mixed runner
- `--host-ram-pressure-limit-pct 90`
- `--host-ram-resume-pct 85`
- `--gpu-memory-pressure-limit-pct 90`
- `--gpu-memory-resume-pct 85`
- `--swap-pressure-limit-pct 100`
- `--swap-resume-pct 100`
- `--pressure-poll-interval-sec 0.5`
- `--pressure-settle-sec 30`
- `--batch-size 186240`
- `--num-workers 0`
- `--repeat-count 5`
- `--width-depths 1,2,3,4,5,6`
- `--missing-present-task-repeats 2,3,4,5`
- `--prediction-repeats 1,2,3,4,5`
- `--gpu-device-index 0`

The CPU-only launcher is the default because it is the safer fallback on this
laptop. The mixed launcher is the one to use when you want the GPU admission
path active again.

Every pressure-aware child also writes `_child_process.log` in its child root.
That file is used by the scheduler to recognize CUDA OOM and
`CUBLAS_STATUS_ALLOC_FAILED` exits when deciding whether to evict a GPU peer
and immediately requeue the failed child.

The old probe path is still available if you want a fixed-slot concurrency
number for a specific laptop. The probe starts at `N=2`, launches the `N`
largest parameter-count candidates first, runs each of them for exactly two
epochs through the normal checkpoint/resume path, and increases `N` until a
trial fails. The last successful `N` is the fixed concurrency to use for the
legacy scheduler.

The probe writes two files into its run root:

- `recommended_parallelism.txt`
- `parallelism_probe_summary.json`

The real launcher accepts `--concurrency-file` and can read the discovered
value directly.

If you want to inspect the full candidate layout before running anything,
generate the per-task plan report first. It writes:

- `planned_params_by_task_depth_width.csv`
- `planned_target_samples_by_task_depth_width.csv`
- per-task `planned_target_samples.csv`
- per-task `planned_candidate_families.csv`
- one plot per task showing the decade distribution and depth coverage

Example planner command:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

./.venv/bin/python MLPS/tabular/shared/dae_dnn/generate_stl_planned_params_report.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --output-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/analysis/planned_params \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 1 3 \
  --num-workers 0 \
  --batch-size 186240
```

The planner reports every sampled target per depth and the deduped candidate
families that the actual run will execute. Use it when you want to see the
exact candidate spread before launching the banded run.

Example optional probe command:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_parallelism_probe.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/parallelism_probe/param_10pow01_03 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 1 3 \
  --probe-epochs 2 \
  --start-n 2 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 186240
```

Example real run command using the pressure-aware scheduler:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 4 6 \
  --repeat-count 5 \
  --scheduler pressure_aware \
  --host-ram-pressure-limit-pct 90 \
  --host-ram-resume-pct 85 \
  --gpu-memory-pressure-limit-pct 90 \
  --gpu-memory-resume-pct 85 \
  --gpu-device-index 0 \
  --max-active-jobs 0 \
  --pressure-settle-sec 30 \
  --max-epochs 100000000 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 186240
```

Example legacy fixed-slot run using the probe output:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 4 6 \
  --concurrency-file MLPS/tabular/shared/dae_dnn/results/stl/parallelism_probe/param_10pow01_03/recommended_parallelism.txt \
  --repeat-count 5 \
  --max-epochs 100000000 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 186240
```

Example for one band:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 4 6 \
  --repeat-count 5 \
  --concurrency 7 \
  --pressure-settle-sec 30 \
  --max-epochs 100000000 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 186240
```

Repeat the same command on the other laptops, changing only the parameter band
and the `--run-root` suffix.

The pressure-aware scheduler runs concrete children globally from
smallest-to-largest and prefers GPU children first while GPU memory is still
below the resume threshold. It may temporarily pause the largest one if host
memory pressure spikes. The probe uses the opposite order and is intentionally
adversarial: it stress tests the heaviest models first.

After all bands finish, merge them into the canonical STL root:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

./.venv/bin/python MLPS/tabular/shared/dae_dnn/merge_stl_ablation_bands.py \
  --input-roots \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow01_03 \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06 \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow07_08 \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow09_10 \
  --output-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1
```

The merge step rebuilds the combined task summaries and plots from the split
band outputs. Keep the band roots as staging inputs until the combined root is
regenerated.

## ADP repeat schedule

The current ADP width-to-depth suite is launcher-driven. Keep the active
`--run-root` aligned with the launcher state and do not reuse a deleted root.
The repeat schedule is encoded in the launcher, not in this document.

Current live ADP repeat split:

- `classification`, `autoencoding`, `generation`, `denoising`, `anomaly`: 4 repeats in the active suite
- `simulation`, `prediction`: 5 repeats in the active suite
- the first five tasks are intended to be merged later with the prior one-off W2D history so each task has 5 combined repeats in the canonical combined tree

The older one-off ADP W2D outputs for the first five tasks are intended to be
merged into the current canonical base later, so the combined history stays
under one task family instead of being split across unrelated roots.

Legacy and analysis helpers:

- `MLPS/tabular/shared/dae_dnn/run_goliath.py`
- `MLPS/tabular/shared/dae_dnn/generate_loss_vs_params_plots.py`
- `MLPS/tabular/shared/dae_dnn/generate_width_only_depth_sweep_plots.py`
- `MLPS/tabular/shared/dae_dnn/generate_w2d_trajectory_loglog_plots.py`
- `MLPS/tabular/shared/dae_dnn/recover_trial1_w2d_history_from_git.py`

## Current training style

The repo uses the same broad method pattern across the tabular sweeps:

1. build the task from `tasks.py`
2. launch a runner entry point
3. write run metadata, logs, CSVs, and phase/task state into the run root
4. restore from the saved checkpoint/state files when resuming
5. keep the repeat-level results separate from the derived analysis outputs
6. probe parameter-band parallelism before launching the real massive STL band

Fresh full strict `4-6` rerun command:

```bash
./MLPS/tabular/shared/dae_dnn/run_stl_massive_band_04_06_fresh.sh
```

This wrapper writes into
`MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06_fresh_v1`
and is intended for a full clean rerun of the strict `10^4..10^6` band. Under
the current generator logic, that band expands to `1928` child architecture
jobs and `9640` repeat phases total because each child runs `--repeat-count 5`.

The current result roots are:

- STL: `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- ADP: `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- archive: `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`

## Resume rule

- same `--run-root` means resume
- new `--run-root` means fresh run
- do not change task order, candidate grid, or repeat count while resuming
- keep the local `main` branch authoritative

## Export rule for other laptops

Non-master laptops should export only generated experiment data:

- `results/`
- `runs/`
- `logs/`
- `metrics/`
- `plots/`
- `checkpoints/`
- `tensorboard/`
- CSV and JSON outputs
- generated figures

Do not merge code changes from those machines into canonical `main`.

## Recommended refresh commands

Refresh the inventory after new results land:

```bash
./.venv/bin/python scripts/update_experiment_inventory.py
```

## Result organization

The canonical result layout is documented in:

- `MLPS/tabular/shared/dae_dnn/results/catalog/`
- `MLPS/tabular/shared/dae_dnn/results/README.md`
- `docs/tabular_dae_dnn/README.md`
