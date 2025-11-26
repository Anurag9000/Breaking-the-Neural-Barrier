# Supervised DAEs (ADP-enabled)

Contains denoising autoencoders trained with paired clean targets.

Current contents:
- `Models/adp_dae_supervised.py`: ADP-enabled Conv DAE (width/depth mutations).
- `run_adp_dae_supervised.py`: runner with ADP policy selector and Gaussian corruption flag.

Quick start:
```bash
python DAE/Supervised/run_adp_dae_supervised.py --policy width_only --corruption-std 0.1 --epochs-per-step 2
```
