# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/denoising_d1`
- Git commit: `c94df6d10754195394b5d5cf6977de976b506d7c`
- Device: `cpu`
- Tasks completed: `['denoising']`

## Task: denoising
- Overall winner: `adp` via `ae_width_only` at `0.000284`
- Winner ADP architecture: `[56]`
- Winner STL architecture: `[56]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [56] | 0.000284 | [56] | 0.023786 | adp |
