# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/representation_d9`
- Git commit: `08c02fe1ed9b88e09a1a3345f5d487222dc92511`
- Device: `cuda`
- Tasks completed: `['representation']`

## Task: representation
- Overall winner: `adp` via `ae_width_only` at `0.084253`
- Winner ADP architecture: `[81, 81, 81, 81, 81, 80, 80, 80, 80]`
- Winner STL architecture: `[81, 81, 81, 81, 81, 80, 80, 80, 80]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [81, 81, 81, 81, 81, 80, 80, 80, 80] | 0.084253 | [81, 81, 81, 81, 81, 80, 80, 80, 80] | 0.167293 | adp |
