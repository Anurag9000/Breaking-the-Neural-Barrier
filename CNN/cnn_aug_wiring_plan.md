# CNN Aug/Normalize Wiring Plan (stepwise)

Goal: wire a shared CIFAR augmentation + normalization pipeline (`utils/cnn_data.py`) into CNN run scripts. Each step lists the target files to patch in order.

## 0) Shared helper (already added)
- `utils/cnn_data.py`: `make_cifar_transforms(dataset, use_augment=True)` with RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize (dataset-specific).

## 1) Core runners (done)
- `CNN/Supervised/Models/CNN_STL_adp_width_to_depth.py` (ADP loader) — aug on by default, `--no-augment` toggle, `--data-root`.
- `CNN/Supervised/Runs/run_cnn_stl.py` — uses shared helper, `--no-augment`.
- `CNN/Supervised/Runs/run_cnn_stl_grid.py` — uses shared helper, `--no-augment`.

## 2) Supervised runners batch 1 (done)
- `run_cnn_alexnet.py`
- `run_cnn_allcnn.py`
- `run_cnn_bam_resnet.py`
- `run_cnn_cbam_resnet.py`
- `run_cnn_convnext.py`
- `run_cnn_cspnet.py`
- `run_cnn_darknet_53.py`
- `run_cnn_densenet.py`
- `run_cnn_dpn.py`
- `run_cnn_eca_resnet.py`
Patch pattern: import `make_cifar_transforms`, add `--no-augment` flag, replace inline transforms with helper (train/eval), keep existing data_root/val split logic.

## 3) Supervised runners batch 2 (done)
- `run_cnn_gc_resnet.py`
- `run_cnn_ghostnet.py`
- `run_cnn_inception_resnet.py`
- `run_cnn_inception_v_1.py`
- `run_cnn_inception_v_3.py`
- `run_cnn_inception_v_4.py`
- `run_cnn_lcl.py`
- `run_cnn_lenet_5.py`
- `run_cnn_mnasnet.py`
- `run_cnn_mobilenet_v_1.py`
Same patch pattern as batch 1.

## 4) Supervised runners batch 3 (done)
- `run_cnn_mobilenet_v_2.py`
- `run_cnn_mobilenet_v_3.py`
- `run_cnn_nfnet.py`
- `run_cnn_nin.py`
- `run_cnn_regnetx.py`
- `run_cnn_replknet.py`
- `run_cnn_repvgg.py`
- `run_cnn_resnet_d.py`
- `run_cnn_resnet_v_1.py`
- `run_cnn_resnet_v_2.py`
Same patch pattern as batch 1.

## 5) Supervised runners batch 4 (done)
- `run_cnn_resnext.py`
- `run_cnn_se_resnet.py`
- `run_cnn_shufflenet_v_1.py`
- `run_cnn_shufflenet_v_2.py`
- `run_cnn_sknet.py`
- `run_cnn_sparsenet.py`
- `run_cnn_squeezenet.py`
- `run_cnn_vgg.py`
- `run_cnn_wideresnet.py`
- Any other supervised `run_cnn_*.py` missed earlier.
Same patch pattern as batch 1.

## 6) Self-supervised runners (reviewed, left unchanged)
These already have strong SSL-specific augmentations and their own normalization/eval pipelines; we do **not** override
them with the CIFAR crop/flip helper to avoid breaking contrastive pretraining. Reviewed scripts:
- `run_barlowtwins_cnn.py`
- `run_byol_cnn.py`
- `run_deepcluster_cnn.py`
- `run_mocov_2_cnn.py`
- `run_pirl_cnn.py`
- `run_rotnet_cnn.py`
- `run_simclr_cnn.py`
- `run_simsiam_cnn.py`
- `run_swav_cnn.py`
- `run_vicreg_cnn.py`
Action: keep SSL augmentations and eval transforms as implemented in each method; no shared helper wiring needed here.

## Notes
- Each patched runner should accept `--no-augment` to disable crop/flip while retaining normalization.
- Respect existing dataset choices (CIFAR-10/100) and data_root/val split logic.
- Avoid touching model definitions; only data/transforms wiring.
