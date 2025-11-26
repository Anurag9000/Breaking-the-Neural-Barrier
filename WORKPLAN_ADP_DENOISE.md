# ADP Denoising Rollout Plan

Status legend: `[ ]` pending, `[~]` in progress, `[x]` done. Remove items when finished.

## 1) Diffusion – Self-Supervised (`Diffusion/Self-Supervised/Models`)
- [x] Add ADP policies (width-only, depth-only, width→depth, depth→width, alt-width-first, alt-depth-first) into `adp_diff_ssl_core.py`.
- [x] Provide runner with flags to pick ADP policy, dataset root, batch size, timesteps, and output dir. (see `Diffusion/Self-Supervised/Runs/adp/run_adp_diff_ssl_*.py`)
- [~] Wire checkpoints/results layout (e.g., `runs/diff_ssl/<algo>/<policy>/`), plus minimal README snippet. (README added in `Diffusion/Self-Supervised/Runs/adp/README.md`; checkpoint path still to wire.)
- [ ] Smoke test on a tiny config (few steps, CPU fallback).

## 2) Diffusion – Supervised (if needed later)
- [ ] Mirror the ADP policies into supervised diffusion cores that currently lack them.
- [ ] Runner + docs for supervised denoising use case.

## 3) Autoencoder – Supervised (`Autoencoder/Supervised/Models`)
- [~] Add ADP width/depth mutations to `ae_denoise_stl.py` (and any parallel AE denoise variants).
- [~] Runner with flags for corruption std, ADP policy, pooling schedule, results dir.
- [ ] Quick validation run (small STL/CIFAR subset).

## 4) Autoencoder – Self-Supervised (`Autoencoder/Self-Supervised/Models`)
- [~] Add denoising-centric modes to the ADP core (e.g., `noise_cond_dae`, `residual_denoise`, `noise_ramp`, `rand_smooth`) with flags.
- [~] Ensure corruption paths stay single-model (no teacher nets).
- [~] Runner with algo selector + result saving.

## 5) Baseline Non-ADP Models
- [ ] Ensure plain denoising AEs (supervised + SSL) remain runnable; add quick-start notes.

## 6) Documentation
- [ ] Add top-level README section linking each runner, key flags, and expected output structure.
- [ ] Note hardware/test settings (CPU/quick smoke vs. full GPU).

## 7) Verification
- [ ] Minimal automated check (imports + forward + one training step) per new runner.
- [ ] Record where artifacts are written; confirm paths exist.
