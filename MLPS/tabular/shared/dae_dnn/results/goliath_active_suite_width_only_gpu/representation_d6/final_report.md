# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/representation_d6`
- Git commit: `e2f333b29b4390437a50baf88415a6d9351067cb`
- Device: `cuda`
- Tasks completed: `['representation']`

## Task: representation
- Overall winner: `adp` via `ae_width_only` at `0.087876`
- Winner ADP architecture: `[105, 105, 105, 105, 105, 105]`
- Winner STL architecture: `[105, 105, 105, 105, 105, 105]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [105, 105, 105, 105, 105, 105] | 0.087876 | [105, 105, 105, 105, 105, 105] | 0.160208 | adp |
