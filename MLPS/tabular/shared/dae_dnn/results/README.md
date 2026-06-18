# Results Layout

The repo now has two distinct views of results:

- a curated catalog under `MLPS/tabular/shared/dae_dnn/results/catalog/`
- the current resumable run tree under `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`

The catalog is the repo-facing organization layer for historical and supporting
results. Keep any active run tree separate from the catalog while it is still
in progress.

This is the canonical layout rule for all MLPS tabular DAE/DNN results in this
repo. Future outputs must be written into the task-first tree and then
published on `main` only.

The canonical result layout for current and future tabular DAE/DNN runs is:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`
- `MLPS/tabular/shared/dae_dnn/results/catalog/<task>/{w2d,width_only,stl_ablation}/`

Use the active suite roots for resumable runs. Put old runs under `archive/`
instead of adding more sibling result trees at the top level.

Current recommended fresh roots:

- STL: `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1`
- small-grid STL: `MLPS/tabular/shared/dae_dnn/results/stl/small_grid/<task>/`
- ADP: `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`

If an ADP suite is partially complete, resume it with:

- `MLPS/tabular/shared/dae_dnn/run_adp_w2d_suite_parallel.py`

That launcher is repeat-ordered. It finishes the current repeat before moving
to the next one.

The preflight candidate planner writes to:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/analysis/planned_params`

That planner emits the combined `planned_params_by_task_depth_width.csv`, the
raw sampled target CSV, and one per-task plot plus per-task CSVs.

Massive STL split runs may stage under parameter-band roots such as:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow01_03`
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06`
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow07_08`
- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow09_10`

Merge those staged roots back into the canonical STL root with:

```bash
./.venv/bin/python MLPS/tabular/shared/dae_dnn/merge_stl_ablation_bands.py \
  --input-roots \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow01_03 \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06 \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow07_08 \
    MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow09_10 \
  --output-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1
```

Before launching any of those parameter-band roots on a given laptop, prefer
the pressure-aware scheduler in
`MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py`. In its default
mode it expands all concrete STL child runs for the chosen tasks/band, sorts
them globally smallest-to-largest by parameter count, but partial child roots
with existing resume state are always considered before untouched jobs. It
fills GPU first up to the GPU memory threshold and then spills extra children
onto CPU while host RAM is still below its resume threshold. After the initial
fill, new launches are completion-gated: a memory-pressure pause or retryable
child failure closes the admission window immediately. The scheduler then
resumes only after a true child completion.

The tabular loaders now also cap any oversize batch to the dataset length, so
when a launcher asks for a batch larger than the split itself, that epoch is a
single batch by construction rather than an implied multi-batch pass.

Runtime tuning for this tabular family is centralized:

- shell launchers source `MLPS/tabular/shared/dae_dnn/runtime_tuning.sh`
- Python entrypoints call `bootstrap_runtime()` before they build tasks or
  enter the main loop; the top-level process re-execs itself under
  `systemd-run --user --scope` in `app-mlps-training.slice` when supported
- fan-out launchers set `TABULAR_CPU_JOB_CONCURRENCY` for each child and
  assign a deterministic `TABULAR_CPU_AFFINITY_CPUS` slice so the child
  process only claims a bounded share of CPU threads and CPUs
- concurrent launchers allocate explicit slot indices for active children so
  simultaneously active jobs get disjoint CPU partitions rather than hashed
  best-effort placement
- `--num-workers 0` means auto-max for these runners, not disabled workers
- loader workers default to the full logical CPU count, and the loaders use
  persistent workers plus prefetching when workers are enabled
- the process attempts best-effort `renice -20` and `ionice -c2 -n0`, and it
  also enables `OMP_WAIT_POLICY=ACTIVE`, `OMP_PROC_BIND=spread`, and
  `OMP_PLACES=cores`; if the OS refuses, the thread, affinity, and worker
  settings still apply
- both bootstraps also request `SCHED_BATCH` so long-running CPU-bound jobs
  are treated as throughput work rather than interactive foreground work
- `MLPS/tabular/shared/dae_dnn/install_linux_runtime_priority.sh` installs the
  persistent host-side settings for `sched_autogroup`, `user-UID.slice`
  weights, and `user@.service` controller delegation
- `MLPS/tabular/shared/dae_dnn/smoke_runtime_priority.py` validates the full
  runtime path under load and checks scope placement, scheduler class,
  affinity partitioning, and aggregate CPU utilization

Scratch outputs from smoke tests, dry runs, recovery probes, and other
validation launches are intentionally excluded from the canonical results tree.
Use them to verify behavior, then delete the generated `results/recovery/*`,
`*_plan.json`, and similar scratch artifacts before publishing a results
snapshot.

The pressure-aware recovery runner does not globally freeze after a single
paused child. Requeued children return to the pending queue and the launcher
keeps admitting other eligible work as long as pressure thresholds allow it.
If a child has already been GPU-paused once, its retry is forced onto CPU so
it does not keep thrashing the same GPU slot.
For this recovery runner, `--max-active-jobs 0` means an automatic bounded
CPU-lane cap, not an unbounded job count. On the 20-core laptop it selects
two active lanes by default, so fallback children get disjoint 10-core
affinity slices instead of one-core slices or overlapping CPU claims. Override
with `TABULAR_RECOVERY_AUTO_ACTIVE_JOBS=<n>` or an explicit
`--max-active-jobs <n>` only when the machine can sustain it.

This is intentionally aggressive. It is meant to keep the CPU side of the
tabular runs busy when the workload can use the extra parallelism.

waits for a true child completion before it considers another launch attempt,
and that next attempt resumes paused or partial work before it admits
untouched jobs. If host RAM pressure exceeds the limit, it requests a pause
on the largest active child, terminates only that child process group, and
later relaunches the same child root so the run resumes from its normal STL
checkpoints. If a child exits with CUDA OOM or `CUBLAS_STATUS_ALLOC_FAILED`,
the scheduler requests a pause on the largest active GPU child, requeues the
failed child at the front of the queue, and retries it again when resources
free up. For the slower laptop split, set `--pressure-settle-sec 120` so
each launch gets a two-minute settle window.

Relevant flags:

- `--scheduler pressure_aware`
- `--host-ram-pressure-limit-pct`
- `--host-ram-resume-pct`
- `--gpu-memory-pressure-limit-pct`
- `--gpu-memory-resume-pct`
- `--gpu-device-index`
- `--max-active-jobs`
- `--max-retries-per-job` as a legacy compatibility flag; pressure-aware mode requeues failed children indefinitely
- `--pressure-poll-interval-sec`
- `--pressure-settle-sec`

For a fresh full strict-band rerun, the repo now includes:

```bash
./MLPS/tabular/shared/dae_dnn/run_stl_massive_band_04_06_fresh.sh
```

That wrapper targets:

```text
MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_06_fresh_v1
```

and launches the whole strict `10^4..10^6` band with:

- all 7 tabular tasks
- `5` repeats per child architecture
- the pressure-aware GPU-first / CPU-spill scheduler
- `--max-active-jobs 0`
- `--batch-size 186240`

Under the current parameter-matched generator, that strict `4-6` band expands
to `1928` child architecture jobs, which means `9640` repeat phases after the
five-repeat expansion inside each child run.

Each pressure-aware child also writes `_child_process.log` in its child root.
That file is used to detect CUDA OOM and cuBLAS allocation failures during
GPU-specific recovery.

The parallelism probe is still available as an optional fixed-slot fallback.
It starts at `N=2`, launches the `N` largest parameter-count candidates
first, runs them for exactly two epochs, and stops when a trial fails. The
last successful `N` becomes the recommended concurrency for a legacy
fixed-slot run.

The real massive STL ablation is not supposed to inherit that short epoch cap.
It should train until early stopping, with the real launcher default now set
to `--max-epochs 100000000` so patience, not a small hard epoch ceiling,
decides when each candidate stops.

Probe outputs to keep:

- `recommended_parallelism.txt`
- `parallelism_probe_summary.json`

The real launcher still accepts `--concurrency-file` for the legacy
fixed-slot path, but the pressure-aware path does not require it. Concrete
STL children are scheduled smallest-to-largest across the whole band; the
probe remains largest-to-smallest because it is intentionally adversarial.

The ADP W2D suite uses task-specific repeat counts. Keep the current launcher
configuration in sync with the active run root and do not reuse a deleted root.

Current live ADP repeat split:

- `classification`, `autoencoding`, `generation`, `denoising`, `anomaly`: 4 repeats in the active suite
- `simulation`, `prediction`: 5 repeats in the active suite
- the first five tasks are intended to be merged later with the older one-off W2D history so each task has 5 combined repeats in the canonical combined tree

Historical ADP W2D archives restored into the repo:

- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_anomaly_onward_gpu`
- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_staged_current`

The staged legacy archive now exposes `classification` at the repo-visible root.

Catalog view:

- `MLPS/tabular/shared/dae_dnn/results/catalog/classification`
- `MLPS/tabular/shared/dae_dnn/results/catalog/autoencoding`
- `MLPS/tabular/shared/dae_dnn/results/catalog/generation`
- `MLPS/tabular/shared/dae_dnn/results/catalog/denoising`
- `MLPS/tabular/shared/dae_dnn/results/catalog/anomaly`
- `MLPS/tabular/shared/dae_dnn/results/catalog/simulation`
- `MLPS/tabular/shared/dae_dnn/results/catalog/prediction`

Each catalog task root mirrors the same three-mode layout:

- `w2d/`
- `width_only/`
- `stl_ablation/`

Enforce this exact structure for every new MLPS results drop:

- task folder at the top level
- `w2d/`, `width_only/`, and `stl_ablation/` directly under the task folder
- depth folders under `width_only/` and `stl_ablation/`
- width folders under each depth folder
- aggregate rollups under `analysis/`
- no alternate top-level result branch for the same run family
- publish the final state on `main` only

`simulation` and `prediction` small-grid STL ablations are now recovered and
assimilated into the canonical task-first tree. Their suite-level provenance is
kept under `results/stl/small_grid/analysis/simulation_prediction_v1`.

Recovered small-grid coverage today:

- `classification`: task-first archive present; STL ablation and W2D recovered; width-only is a placeholder
- `autoencoding`: task-first archive present; STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `generation`: task-first archive present; STL ablation, W2D, and width-only recovered for depths `d1` through `d6`
- `denoising`: task-first archive present; STL ablation, W2D, and width-only recovered for depths `d1` through `d10`
- `anomaly`: task-first archive present; STL ablation and W2D recovered; width-only is a placeholder
- `simulation`: task-first archive present; STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only are placeholders
- `prediction`: task-first archive present; STL ablation recovered for depths `d03`, `d04`, `d06`, `d08`, `d10`; W2D and width-only are placeholders

STL ablation is still a flagged TODO in the repo docs. Do not treat the
archived STL tree as finished state; it is the history backing the current
fresh run layout and the remaining analysis work.

The planned STL schedule lives at:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/planned_params_by_task_depth_width.csv`

Archived outputs already moved there include:

- `MLPS/tabular/shared/dae_dnn/results/archive/stl_ablation_parameter_matched_gpu_serial`
- `MLPS/tabular/shared/dae_dnn/results/archive/classification_trial1`

`classification_trial1` is the older lightweight STL study, not the massive
all-task ablation. It covers:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`

It does not include dedicated STL roots for `simulation` or `prediction`.

Experiment catalog files:

- `docs/tabular_dae_dnn/experiment_inventory.md`
- `docs/tabular_dae_dnn/experiment_inventory.csv`

Regenerate the catalog with:

```bash
./.venv/bin/python scripts/update_experiment_inventory.py
```
