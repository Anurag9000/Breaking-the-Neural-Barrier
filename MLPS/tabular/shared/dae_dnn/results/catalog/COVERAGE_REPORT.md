# Tabular DAE/DNN Results Coverage

This catalog is the task-first organization layer for scattered historical
results. It links to the original payload roots instead of duplicating logs,
CSVs, JSON, plots, or checkpoint-adjacent metadata.

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
| anomaly | 4 complete + 1 partial | repeats 1-4 complete; old `anomaly_onward_gpu` W2D candidate is linked under `w2d/partial/` and has `completed=false` |
| simulation | 5 | repeats 1-4 from `main_without_denoising_v1`; repeat 5 from `denoising_plus_sim5_v1` |
| prediction | 5 | repeats 1-4 from `main_without_denoising_v1`; repeat 5 from `denoising_plus_sim5_v1` |

Missing for a strict 5-complete W2D set:

- `anomaly` repeat 5 completed state

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
| anomaly | 20 | missing `d06/w256`, all `d08/*`, and all `d10/*` |
| simulation | 35 | candidate metadata/log/CSV payloads present |
| prediction | 35 | candidate metadata/log/CSV payloads present |

Missing for a strict full small-grid archive:

- `anomaly`: 15 depth/width combinations
- any five-repeat interpretation: no task in `small_grid` has five explicit repeats

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

