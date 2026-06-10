# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/denoising_d4`
- Git commit: `c94df6d10754195394b5d5cf6977de976b506d7c`
- Device: `cpu`
- Tasks completed: `['denoising']`

## Task: denoising
- Overall winner: `adp` via `ae_width_only` at `0.000721`
- Winner ADP architecture: `[121, 121, 121, 121]`
- Winner STL architecture: `[121, 121, 121, 121]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [121, 121, 121, 121] | 0.000721 | [121, 121, 121, 121] | 0.010436 | adp |
