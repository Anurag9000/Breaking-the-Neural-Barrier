# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/simulation_d3`
- Git commit: `179b42b51337382feb7be02f91979d14aacbe698`
- Device: `cpu`
- Tasks completed: `['simulation']`

## Task: simulation
- Overall winner: `adp` via `ae_width_only` at `0.000768`
- Winner ADP architecture: `[56, 56, 56]`
- Winner STL architecture: `[56, 56, 56]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [56, 56, 56] | 0.000768 | [56, 56, 56] | 0.004797 | adp |
