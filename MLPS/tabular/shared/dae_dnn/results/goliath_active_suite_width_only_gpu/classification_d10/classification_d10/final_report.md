# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/representation_d10`
- Git commit: `08c02fe1ed9b88e09a1a3345f5d487222dc92511`
- Device: `cuda`
- Tasks completed: `['representation']`

## Task: representation
- Overall winner: `adp` via `ae_width_only` at `0.083482`
- Winner ADP architecture: `[79, 79, 79, 79, 79, 79, 79, 79, 79, 78]`
- Winner STL architecture: `[79, 79, 79, 79, 79, 79, 79, 79, 79, 78]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [79, 79, 79, 79, 79, 79, 79, 79, 79, 78] | 0.083482 | [79, 79, 79, 79, 79, 79, 79, 79, 79, 78] | 0.167884 | adp |
