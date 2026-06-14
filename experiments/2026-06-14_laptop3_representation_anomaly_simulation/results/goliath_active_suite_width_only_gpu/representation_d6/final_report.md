# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/representation_d6`
- Git commit: `179b42b51337382feb7be02f91979d14aacbe698`
- Device: `cpu`
- Tasks completed: `['representation']`

## Task: representation
- Overall winner: `adp` via `ae_width_only` at `0.086212`
- Winner ADP architecture: `[101, 101, 101, 101, 100, 100]`
- Winner STL architecture: `[101, 101, 101, 101, 100, 100]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [101, 101, 101, 101, 100, 100] | 0.086212 | [101, 101, 101, 101, 100, 100] | 0.185047 | adp |
