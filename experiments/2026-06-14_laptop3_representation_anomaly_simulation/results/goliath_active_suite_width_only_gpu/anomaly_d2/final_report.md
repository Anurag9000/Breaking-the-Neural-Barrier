# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/anomaly_d2`
- Git commit: `179b42b51337382feb7be02f91979d14aacbe698`
- Device: `cpu`
- Tasks completed: `['anomaly']`

## Task: anomaly
- Overall winner: `adp` via `ae_width_only` at `0.000668`
- Winner ADP architecture: `[87, 87]`
- Winner STL architecture: `[87, 87]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [87, 87] | 0.000668 | [87, 87] | 0.002749 | adp |
