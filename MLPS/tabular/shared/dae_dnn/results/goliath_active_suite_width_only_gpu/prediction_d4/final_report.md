# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/prediction_d4`
- Git commit: `179b42b51337382feb7be02f91979d14aacbe698`
- Device: `cpu`
- Tasks completed: `['prediction']`

## Task: prediction
- Overall winner: `adp` via `ae_width_only` at `0.251503`
- Winner ADP architecture: `[58, 58, 57, 57]`
- Winner STL architecture: `[58, 58, 57, 57]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [58, 58, 57, 57] | 0.251503 | [58, 58, 57, 57] | 0.279220 | adp |
