# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/anomaly_d1`
- Git commit: `179b42b51337382feb7be02f91979d14aacbe698`
- Device: `cpu`
- Tasks completed: `['anomaly']`

## Task: anomaly
- Overall winner: `adp` via `ae_width_only` at `0.000662`
- Winner ADP architecture: `[114]`
- Winner STL architecture: `[114]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [114] | 0.000662 | [114] | 0.001831 | adp |
