# DAE Collection (Supervised & Self-Supervised)

This directory houses denoising autoencoders with ADP (Adaptive Depth/Width) variants.

Layout:
- `Supervised/` — DAE models trained with paired clean targets.
- `Unsupervised/` — Self-supervised DAEs with various corruptions (noise, masking, inpainting, blind-spot, energy).

Each subfolder will include:
- Base model definitions.
- ADP wrappers (width/depth mutations).
- Runners (small configs for smoke tests).
