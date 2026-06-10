# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/generation_d4`
- Git commit: `c8ccd9f2a922d72f9f7dddf1a2c721cf7c21c907`
- Device: `cpu`
- Tasks completed: `['generation']`

## Task: generation
- Overall winner: `adp` via `ae_width_only` at `0.000302`
- Winner ADP architecture: `[69, 69, 68, 68]`
- Winner STL architecture: `[69, 69, 68, 68]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [69, 69, 68, 68] | 0.000302 | [69, 69, 68, 68] | 1.008313 | adp |
