# Tabular DAE/DNN Results Coverage

This catalog is the task-first organization layer for scattered historical
results. It links to the original payload roots instead of duplicating logs,
CSVs, JSON, plots, or checkpoint-adjacent metadata.

Current missing inventory at a glance:

| Family | Missing now |
| --- | --- |
| ADP W2D | none at the catalog level; anomaly repeat 5 is cataloged via operator declaration, while the raw source remains a partial trace |
| ADP width-only d01-d06 | `prediction` is missing entirely; no recovered width-only export exposes an explicit five-repeat layout, so normalized repeat slots 2-5 are absent for every present task/depth |
| Small STL grid | `anomaly` is missing `d06/w256`, all `d08/*`, and all `d10/*`; no small-grid task has an explicit five-repeat structure |
| Massive STL ablation | staged bands `param_10pow01_03` and `param_10pow07_08` are absent |

Recovery runner:

- `MLPS/tabular/shared/dae_dnn/run_missing_width_stl_recovery_pressure.sh`
- `MLPS/tabular/shared/dae_dnn/run_missing_width_stl_recovery_pressure_gpu_cpu.sh`
- writes to `MLPS/tabular/shared/dae_dnn/results/recovery/missing_width_stl_v1`
- the default wrapper is CPU-only; the mixed wrapper keeps CUDA visible
- queues the width-only repeat/depth gaps and the missing anomaly small-grid
  leaves together, sorted by estimated parameter count
- the mixed wrapper uses GPU first while GPU memory is under the resume
  threshold, then CPU while host RAM is under the resume threshold
- pauses the largest active child on GPU or host RAM pressure, requeues the same
  child root, and waits for a true child completion before admitting more work

Task folders:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`

Each task keeps:

- `w2d/`
- `width_only/`
- `stl_ablation/`

## ADP W2D

Canonical combined view:

- `results/catalog/<task>/w2d/combined_5repeat/`

Coverage:

| Task | Completed W2D runs found | Notes |
| --- | ---: | --- |
| classification | 5 | repeats 1-4 from `main_without_denoising_v1`; repeat 5 is the legacy `representation` archive exposed under classification |
| autoencoding | 5 | repeats 1-4 from `main_without_denoising_v1`; repeat 5 from `archive/goliath_w2d_staged_current` |
| generation | 5 | repeats 1-4 from `main_without_denoising_v1`; repeat 5 from `archive/goliath_w2d_staged_current` |
| denoising | 5 | repeats 1-4 from `denoising_plus_sim5_v1`; repeat 5 from `archive/goliath_w2d_staged_current` |
| anomaly | 5 cataloged | repeats 1-4 complete; repeat 5 is the old `anomaly_onward_gpu` W2D trace, operator-declared complete and linked into `combined_5repeat/repeat_05_operator_declared`; the raw source remains linked under `w2d/partial/` and still records `completed=false` |
| simulation | 5 | repeats 1-4 from `main_without_denoising_v1`; repeat 5 from `denoising_plus_sim5_v1` |
| prediction | 5 | repeats 1-4 from `main_without_denoising_v1`; repeat 5 from `denoising_plus_sim5_v1` |

Missing for the cataloged 5-repeat W2D set:

- none at the catalog level

Provenance exception:

- `anomaly` repeat 5 is not a natural runner-completed state. It is included
  because the operator explicitly declared the partial trace completed for the
  final results bundle. The raw trace was not rewritten.

## ADP Width-Only Depths 1-6

Canonical combined view:

- `results/catalog/<task>/width_only/by_depth_d01_d06/`

Coverage:

| Task | Depths present | Notes |
| --- | --- | --- |
| classification | d01-d06 | present as legacy `representation_d1` through `representation_d6` |
| autoencoding | d01-d06 | present under `goliath_active_suite_width_only_gpu`; d01-d04 use final reports instead of task-state files |
| generation | d01-d06 | present under `goliath_active_suite_width_only_gpu` |
| denoising | d01-d06 | present in the laptop3 denoising export |
| anomaly | d01-d06 | present in the laptop3 representation/anomaly/simulation export |
| simulation | d01-d06 | present in the laptop3 representation/anomaly/simulation export |
| prediction | missing | no width-only d01-d06 payload found in current tree or reachable git objects |

Important: these width-only exports do not expose a five-repeat layout per
task/depth. They are linked as one completed payload per task/depth where
present. If the experimental requirement is five repeats per depth, the
missing repeat count is all repeats beyond the single recovered payload for
the present tasks, and all six depths for `prediction`.

Search audit for `prediction` width-only:

- current filesystem was searched for `prediction_d1` through `prediction_d6`,
  `prediction/.../ae_width_only`, `prediction/.../stl_from_ae_width_only`, and
  `width_only...prediction`
- full git object history was searched with the same patterns
- only the catalog placeholder path was found; no source payload or historical
  result object for prediction width-only was found

## Small STL Grid

Canonical view:

- `results/catalog/<task>/stl_ablation/small_grid`
- source payloads live at `results/stl/small_grid/<task>/stl_ablation`

Expected small-grid shape in the recovered archive:

- depths: `d03`, `d04`, `d06`, `d08`, `d10`
- widths: `w064`, `w096`, `w128`, `w160`, `w192`, `w224`, `w256`
- no repeat dimension is represented in this archive

Coverage:

| Task | Depth/width folders present | Leaf payload status |
| --- | ---: | --- |
| classification | 35 | folder-level archive present; many leaves are summary/provenance-only |
| autoencoding | 35 | folder-level archive present; many leaves are summary/provenance-only |
| generation | 35 | folder-level archive present; many leaves are summary/provenance-only |
| denoising | 35 | folder-level archive present; many leaves are summary/provenance-only |
| anomaly | 20 | present, but missing `d06/w256`, all `d08/*`, and all `d10/*` |
| simulation | 35 | candidate metadata/log/CSV payloads present |
| prediction | 35 | candidate metadata/log/CSV payloads present |

Missing for a strict full small-grid archive:

- `anomaly`: 15 depth/width combinations
- any five-repeat interpretation: no task in `small_grid` has five explicit repeats

Search audit for `anomaly` small-grid:

- current filesystem and full git object history both show the same recovered
  20 anomaly summary leaves
- no `anomaly` small-grid objects were found for `d08/*`, `d10/*`, or
  `d06/w256`

Exact missing `anomaly` small-grid leaves:

- `d06/w256`
- `d08/w064`
- `d08/w096`
- `d08/w128`
- `d08/w160`
- `d08/w192`
- `d08/w224`
- `d08/w256`
- `d10/w064`
- `d10/w096`
- `d10/w128`
- `d10/w160`
- `d10/w192`
- `d10/w224`
- `d10/w256`

## Massive STL Ablation

Canonical view:

- `results/catalog/<task>/stl_ablation/massive_parammatched_decade_v1`
- source payloads live at `results/stl/ablation/parammatched_decade_v1/<task>`

Staging roots currently present:

- `results/stl/ablation/parammatched_decade_v1_param_10pow04_06`
- `results/stl/ablation/parammatched_decade_v1_param_10pow04_06_cpu`
- `results/stl/ablation/parammatched_decade_v1_param_10pow04_07`

Missing staged band roots:

- `results/stl/ablation/parammatched_decade_v1_param_10pow01_03`
- `results/stl/ablation/parammatched_decade_v1_param_10pow07_08`

Current canonical merged root is present for all seven tasks, but the available
band staging evidence is concentrated in the `10^4..10^6` / `10^4..10^7`
roots. Treat `1..3` and `7..8` as absent unless recovered from another clone.
