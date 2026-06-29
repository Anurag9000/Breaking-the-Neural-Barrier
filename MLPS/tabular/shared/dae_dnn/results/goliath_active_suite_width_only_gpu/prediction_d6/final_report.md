# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/prediction_d6`
- Git commit: `179b42b51337382feb7be02f91979d14aacbe698`
- Device: `cpu`
- Tasks completed: `['prediction']`

## Task: prediction
- Overall winner: `adp` via `ae_width_only` at `0.231796`
- Winner ADP architecture: `[30, 30, 30, 30, 30, 29]`
- Winner STL architecture: `[30, 30, 30, 30, 30, 29]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [30, 30, 30, 30, 30, 29] | 0.231796 | [30, 30, 30, 30, 30, 29] | 0.350522 | adp |
