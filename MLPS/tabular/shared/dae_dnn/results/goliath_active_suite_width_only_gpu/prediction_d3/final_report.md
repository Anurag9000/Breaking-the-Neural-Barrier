# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/prediction_d3`
- Git commit: `179b42b51337382feb7be02f91979d14aacbe698`
- Device: `cpu`
- Tasks completed: `['prediction']`

## Task: prediction
- Overall winner: `adp` via `ae_width_only` at `0.263661`
- Winner ADP architecture: `[45, 45, 44]`
- Winner STL architecture: `[45, 45, 44]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [45, 45, 44] | 0.263661 | [45, 45, 44] | 0.278876 | adp |
