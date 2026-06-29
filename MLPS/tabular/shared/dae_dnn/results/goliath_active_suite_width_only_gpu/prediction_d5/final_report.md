# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/prediction_d5`
- Git commit: `179b42b51337382feb7be02f91979d14aacbe698`
- Device: `cpu`
- Tasks completed: `['prediction']`

## Task: prediction
- Overall winner: `adp` via `ae_width_only` at `0.274865`
- Winner ADP architecture: `[34, 34, 34, 34, 33]`
- Winner STL architecture: `[34, 34, 34, 34, 33]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [34, 34, 34, 34, 33] | 0.274865 | [34, 34, 34, 34, 33] | 0.307385 | adp |
