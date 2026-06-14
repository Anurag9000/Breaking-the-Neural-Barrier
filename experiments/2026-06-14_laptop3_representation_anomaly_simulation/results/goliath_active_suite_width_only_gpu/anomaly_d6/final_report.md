# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/anomaly_d6`
- Git commit: `179b42b51337382feb7be02f91979d14aacbe698`
- Device: `cpu`
- Tasks completed: `['anomaly']`

## Task: anomaly
- Overall winner: `adp` via `ae_width_only` at `0.000888`
- Winner ADP architecture: `[86, 85, 85, 85, 85, 85]`
- Winner STL architecture: `[86, 85, 85, 85, 85, 85]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [86, 85, 85, 85, 85, 85] | 0.000888 | [86, 85, 85, 85, 85, 85] | 0.007397 | adp |
