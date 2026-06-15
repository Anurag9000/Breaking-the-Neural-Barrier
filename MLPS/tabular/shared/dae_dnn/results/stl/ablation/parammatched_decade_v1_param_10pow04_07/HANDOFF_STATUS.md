## STL Ablation Band 10^4..10^7 Handoff

This document records the exact on-disk state of the staged STL ablation band root:

- `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_07`

It was written after pulling `origin/main` through commit `d2a693eabc` (`Harden STL pressure-aware scheduler`) and then restoring the local staged band/probe artifacts.

### Canonical Roots

- Band run root:
  - `MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_07`
- Parallelism probe root for the same band:
  - `MLPS/tabular/shared/dae_dnn/results/stl/parallelism_probe/param_10pow04_07`

### Planned Band Size

The saved probe log for `param_10pow04_07` records the candidate counts used by the current code:

- `classification`: `360`
- `autoencoding`: `334`
- `generation`: `360`
- `denoising`: `360`
- `anomaly`: `361`
- `simulation`: `351`
- `prediction`: `351`

Totals:

- concrete STL candidate families in this band: `2477`
- total repeat-level STL runs at `repeat_count=5`: `12385`

### Confirmed Completed Work

Completed repeat-level STL runs currently present on disk: `202`

- `classification/_children/stl_ablation_r01_d01_parammatched`: `50` complete, `1` partial
- `classification/_children/stl_ablation_r01_d02_parammatched`: `76` complete, `1` partial
- `classification/_children/stl_ablation_r01_d03_parammatched`: `76` complete, `1` partial

No other task has a completed repeat-level STL run in this staged band root yet.

Completed runs that already ended at `final_epoch >= 200` before the epoch-cap lift: `39`

The `d1` child later resumed past the old cap, and the current partial run there has `final_epoch=303` in its saved state.

### Exact Partial Resume Points

These are the three original partially completed classification runs that were already in progress before the later failed restart attempts:

1. `classification/_children/stl_ablation_r01_d01_parammatched/classification/stl_ablation_r01_d3125_parammatched/cand_000_d1_w3125`
   - `epoch=123`
   - `batch_index=21`
   - `epoch_complete=false`
   - `checkpoint_last.pt` and `checkpoint_best.pt` both exist

2. `classification/_children/stl_ablation_r01_d02_parammatched/classification/stl_ablation_r02_d02_w0804_804_804/cand_000_d2_w804`
   - `epoch=125`
   - `batch_index=29`
   - `epoch_complete=false`
   - `checkpoint_last.pt` and `checkpoint_best.pt` both exist

3. `classification/_children/stl_ablation_r01_d03_parammatched/classification/stl_ablation_r02_d03_w0574_574_574_574/cand_000_d3_w574`
   - `epoch=11`
   - `batch_index=21`
   - `epoch_complete=false`
   - `checkpoint_last.pt` and `checkpoint_best.pt` both exist

### Abandoned Later Start Attempts

Later pressure-aware and fixed-concurrency restart attempts created additional child roots. These are incomplete and may be resumed or discarded, but they should not be mistaken for completed work.

Child roots with saved `candidate_state.json` progress:

- `anomaly/_children/stl_ablation_r01_d08_w0029_29_29_29_29_29_29_29_29`
  - `1` partial candidate
  - max saved epoch: `16`
- `anomaly/_children/stl_ablation_r01_d10_w0026_26_26_26_26_26_26_26_26_26_26`
  - `1` partial candidate
  - max saved epoch: `50`
- `autoencoding/_children/stl_ablation_r01_d08_w0029_29_29_29_29_29_29_29_29`
  - `1` partial candidate
  - max saved epoch: `3`
- `autoencoding/_children/stl_ablation_r01_d10_w0026_26_26_26_26_26_26_26_26_26_26`
  - `1` partial candidate
  - max saved epoch: `13`
- `classification/_children/stl_ablation_r01_d06_w0037_37_37_37_37_37_37`
  - `1` partial candidate
  - max saved epoch: `1`
- `denoising/_children/stl_ablation_r01_d08_w0029_29_29_29_29_29_29_29_29`
  - `1` partial candidate
  - max saved epoch: `1`
- `denoising/_children/stl_ablation_r01_d10_w0026_26_26_26_26_26_26_26_26_26_26`
  - `1` partial candidate
  - max saved epoch: `10`
- `generation/_children/stl_ablation_r01_d08_w0029_29_29_29_29_29_29_29_29`
  - `1` partial candidate
  - max saved epoch: `1`
- `generation/_children/stl_ablation_r01_d10_w0026_26_26_26_26_26_26_26_26_26_26`
  - `1` partial candidate
  - max saved epoch: `7`
- `prediction/_children/stl_ablation_r01_d07_w0038_38_38_38_38_38_38_38`
  - `1` partial candidate
  - max saved epoch: `20`

Child roots created but with no saved candidate state yet:

- `anomaly/_children/stl_ablation_r01_d07_w0031_31_31_31_31_31_31_31`
- `autoencoding/_children/stl_ablation_r01_d07_w0031_31_31_31_31_31_31_31`
- `classification/_children/stl_ablation_r01_d05_w0041_41_41_41_41_41`
- `classification/_children/stl_ablation_r01_d08_w0032_32_32_32_32_32_32_32_32`
- `classification/_children/stl_ablation_r01_d09_w0030_30_30_30_30_30_30_30_30_30`
- `denoising/_children/stl_ablation_r01_d07_w0031_31_31_31_31_31_31_31`
- `generation/_children/stl_ablation_r01_d07_w0031_31_31_31_31_31_31_31`
- `prediction/_children/stl_ablation_r01_d04_w0054_54_54_54_54`
- `prediction/_children/stl_ablation_r01_d10_w0031_31_31_31_31_31_31_31_31_31_31`
- `simulation/_children/stl_ablation_r01_d04_w0054_54_54_54_54`
- `simulation/_children/stl_ablation_r01_d07_w0038_38_38_38_38_38_38_38`
- `simulation/_children/stl_ablation_r01_d10_w0031_31_31_31_31_31_31_31_31_31_31`

### Current Totals In This Staged Root

- completed repeat-level runs: `202`
- partial repeat-level runs with `candidate_state.json`: `13`
- remaining repeat-level runs not completed yet: `12183`

Those `12183` remaining runs include both:

- the `13` partial runs above, and
- all band jobs that were never started at all

### Probe State

Probe root:

- `MLPS/tabular/shared/dae_dnn/results/stl/parallelism_probe/param_10pow04_07`

What exists:

- `training_log.txt`
- `probe_n03/`

What does not exist:

- `parallelism_probe_summary.json`
- `recommended_parallelism.txt`
- `probe_n03/trial_state.json`

The probe log shows:

- tasks: `classification autoencoding generation denoising anomaly simulation prediction`
- probe epochs: `2`
- band: `[4, 7]`
- probe candidates: `2477`
- largest candidates:
  - `classification:[3774 x 8] @ 100022329`
  - `anomaly:[4460 x 6] @ 100020014`
  - `autoencoding:[4460 x 6] @ 100020014`
  - `denoising:[4460 x 6] @ 100020014`
  - `generation:[4460 x 6] @ 100020014`
- last saved probe event:
  - `[PROBE] start n=3`

Interpretation:

- the `n=3` probe began
- the machine died before trial completion state was written
- this probe root should be treated as an incomplete failed probe attempt, not a finished recommendation

### Host-Side Memory Pressure Setting Used On This Laptop

The `90%` memory-pressure setting that was tested is an OS configuration on this laptop, not a repo-tracked experiment flag:

- `DefaultMemoryPressureLimit = 90%`
- `ManagedOOMMemoryPressureLimit = 90%`

Those values are not portable through Git by themselves. Another laptop must set its own `systemd-oomd` policy separately if it needs to reproduce the same host behavior.

### Resume Options On Another Laptop

#### Resume From The Exact Saved State

Use the same band root and let the launcher reuse the existing child directories:

```bash
cd /path/to/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1_param_10pow04_07 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current \
  --tasks classification autoencoding generation denoising anomaly simulation prediction \
  --param-band 4 7 \
  --repeat-count 5 \
  --concurrency 1 \
  --max-epochs 100000000 \
  --num-workers 0 \
  --pin-memory \
  --batch-size 9312
```

That command will:

- keep the already completed `202` runs
- reuse the three original partial classification candidates
- reuse or retry the later abandoned child roots if they match the deterministic job list

#### Discard Partial Attempts And Start The Band Fresh

If the abandoned child roots are not worth preserving, keep the completed-count documentation from this file and start a fresh band root on the stronger machine instead:

- keep this staged root as an audit record
- launch a new clean run root for `param_10pow04_07`
- do not rely on these partial child roots for resume

### Practical Interpretation

If you want a fast, conservative continuation later:

- treat the `202` completed repeat-level runs as real completed work
- treat the `13` partial repeat-level runs plus the additional unstarted child roots as optional resume material
- treat the `n=3` probe as failed and non-authoritative

If you do not care about salvaging partials:

- keep this root for accounting only
- restart the entire `10^4..10^7` band on the stronger laptop
- count this machine's contribution as `202` completed runs plus useful failure evidence about instability
