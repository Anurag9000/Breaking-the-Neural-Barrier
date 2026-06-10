# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/generation_d5`
- Git commit: `c8ccd9f2a922d72f9f7dddf1a2c721cf7c21c907`
- Device: `cpu`
- Tasks completed: `['generation']`

## Task: generation
- Overall winner: `adp` via `ae_width_only` at `0.000253`
- Winner ADP architecture: `[108, 108, 108, 108, 108]`
- Winner STL architecture: `[108, 108, 108, 108, 108]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [108, 108, 108, 108, 108] | 0.000253 | [108, 108, 108, 108, 108] | 1.008206 | adp |
