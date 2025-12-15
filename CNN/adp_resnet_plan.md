# ADP-ResNet Plan (Stepwise Execution Log)

Goal: build a stronger ADP pipeline on a ResNet-style CNN backbone for CIFAR-10, with all ADP modes, STL + grid baselines, and optional CutMix/dropout, while keeping the existing ConvNetSTL ADP intact.

This file tracks the steps and what has been implemented.

---

## 1) New backbone and structure

- [x] Create `CNN/ADP_ResNet/` subfolder.
- [x] Implement `adp_resnet_backbone.py`:
  - CIFAR ResNet-style model `ADPResNet` with:
    - 3 stages (C1, C2, C3) with base channels scaled by `width`.
    - Basic residual blocks: `Conv3x3 -> BN -> ReLU -> (Dropout) -> Conv3x3 -> BN`, with identity or 1×1 skip.
    - Configurable `width` (base channel multiplier) and `depth` (blocks per stage).
  - Helper functions:
    - `make_adp_resnet(width: int, depth: int, num_classes: int = 10)`.
    - `estimate_neurons(width, depth)` used by ADP (currently `width * (depth + 1)` as a proxy).
- [x] Keep backbone self-contained and independent from the old ConvNetSTL.

## 2) ADP runner on ADPResNet (all modes)

- [x] Add `CNN/ADP_ResNet/run_adp_resnet.py` which:
  - Uses CIFAR loaders via the shared helper `utils/cnn_data.make_cifar_transforms`.
  - Implements a clean classification-only ADP loop:
    - `ADPConfig` with `adp_mode in {width_only, depth_only, width_to_depth, depth_to_width, alt_width, alt_depth}`.
    - Expands **width** (channels) and **depth** (blocks per stage) while attempting to preserve weights.
    - Tracks global best validation loss and architecture snapshot.
    - Uses early stopping based on validation loss (`delta`, `patience`).
  - Training loop:
    - Loss: cross-entropy.
    - Optimizer: AdamW.
    - LR: CosineAnnealingLR (per config) with configurable `lr`, `min_lr`, and `T_max = max_epochs`.
    - Grad clipping on all parameters.
    - Optional CutMix per batch controlled by `cutmix_p`, `cutmix_alpha` in `ADPConfig`.
  - Logging:
    - Console + CSV (`training_stats.csv` in results dir) via `ContinuousLogger`.
    - Optional plots via `utils/adp_plot` (loss vs epoch, loss vs neurons).
- [x] CLI arguments:
  - Architecture: `--width`, `--depth`, `--max-width`, `--max-depth`, `--max-neurons`, `--dropout`.
  - ADP behaviour: `--adp-mode`, `--trials-width`, `--trials-depth`, `--delta`, `--patience`, `--ex-k`.
  - Training: `--lr`, `--min-lr`, `--weight-decay`, `--grad-clip`, `--max-epochs`.
  - Data: `--dataset` (cifar10|cifar100), `--data-root`, `--batch-size`, `--val-split`, `--num-workers`, `--no-augment`.
  - Regularisation: `--cutmix-p`, `--cutmix-alpha`.
  - Logging/plots: `--results-dir`, `--plot-loss`, `--plot-neurons`.

## 3) Baseline STL + grid STL for ADPResNet

- [x] Add `CNN/ADP_ResNet/run_resnet_stl.py`:
  - Single-model supervised training (no ADP).
  - Reuses `ADPResNet` backbone and the same CIFAR data pipeline as `run_adp_resnet.py`.
  - Training loop:
    - Cross-entropy, AdamW, cosine LR, grad clipping.
    - Optional CutMix (`--cutmix-p`, `--cutmix-alpha`) and configurable dropout.
    - Early stopping on validation loss with `--patience`.
  - Outputs:
    - `stl_summary.json` with config + best metrics.
    - Epoch-vs-loss (log-y) and epoch-vs-accuracy plots in the results directory.
- [x] Add `CNN/ADP_ResNet/run_resnet_stl_grid.py`:
  - Grid over widths and depths (similar spirit to `run_cnn_stl_grid.py` but using ADPResNet).
  - Per-configuration training uses AdamW + CrossEntropy + optional CutMix.
  - Logs each configuration’s best val loss, accuracy, and estimated neurons.
  - Saves CSV/JSON plus combined neurons-vs-loss and neurons-vs-accuracy scatter plots.

## 4) Strong data & regularization pipeline

- [x] Use `utils/cnn_data.make_cifar_transforms` everywhere for CIFAR normalization and base augmentations (crop + flip).
- [x] Add CutMix and dropout options:
  - Implemented `utils/cutmix.py` with `cutmix_batch(x, y, alpha, p)`.
  - Runners (`run_adp_resnet.py`, `run_resnet_stl.py`, `run_resnet_stl_grid.py`) accept:
    - `--cutmix-p`, `--cutmix-alpha` (0 disables).
  - Backbone supports configurable dropout in blocks via `ADPResNetConfig.dropout` and CLI flags.
- [x] Ensure all new runners respect `--no-augment` but keep normalization.

## 5) Verification and usage

- [ ] Sanity-check imports and basic `--help` for each new script.
- [ ] Run a short ADP-ResNet experiment on CIFAR-10 to verify:
  - Loss decreases below ConvNetSTL ADP (aiming for < 1.8 with modest width/depth).
  - Width and depth actually expand as expected per ADP mode.
- [ ] Run a short STL baseline (no ADP) to confirm it reaches significantly better loss/accuracy than the current ConvNetSTL STL.

As steps are implemented, update the checkboxes above so this file stays as the authoritative “what’s done” log for the ADP-ResNet pipeline.

