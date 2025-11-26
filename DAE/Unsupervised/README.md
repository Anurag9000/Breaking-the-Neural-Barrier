# Self-Supervised DAEs (ADP-enabled)

Contains single-model denoising AEs with self-supervised corruptions.

Current contents:
- `Models/adp_dae_unsupervised.py`: ADP-enabled DenoisingConvAE + resize helpers.
- `run_adp_dae_unsupervised.py`: runner with corruption selector (gaussian, pixel_mask, patch_mask, blindspot, inpaint, energy) and ADP policies.

Quick start:
```bash
python DAE/Unsupervised/run_adp_dae_unsupervised.py --policy width_only --mode gaussian --std 0.1 --epochs-per-step 2
```

Additional examples:
- Blind-spot loss on masked pixels: `--mode blindspot --mask-prob 0.05`
- Inpainting (rect holes): `--mode inpaint --holes-per-image 1 --min-hole-frac 0.15 --max-hole-frac 0.35`
- Energy contrast: `--mode energy --energy-neg-mode roll --energy-margin 0.05`
