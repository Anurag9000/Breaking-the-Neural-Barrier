# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/representation_d5`
- Git commit: `179b42b51337382feb7be02f91979d14aacbe698`
- Device: `cpu`
- Tasks completed: `['representation']`

## Task: representation
- Overall winner: `adp` via `ae_width_only` at `0.086734`
- Winner ADP architecture: `[107, 107, 106, 106, 106]`
- Winner STL architecture: `[107, 107, 106, 106, 106]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [107, 107, 106, 106, 106] | 0.086734 | [107, 107, 106, 106, 106] | 0.159695 | adp |
