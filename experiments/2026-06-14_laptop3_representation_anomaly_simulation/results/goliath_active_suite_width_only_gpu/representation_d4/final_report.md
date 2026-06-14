# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/representation_d4`
- Git commit: `179b42b51337382feb7be02f91979d14aacbe698`
- Device: `cpu`
- Tasks completed: `['representation']`

## Task: representation
- Overall winner: `adp` via `ae_width_only` at `0.088354`
- Winner ADP architecture: `[140, 140, 140, 140]`
- Winner STL architecture: `[140, 140, 140, 140]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [140, 140, 140, 140] | 0.088354 | [140, 140, 140, 140] | 0.172570 | adp |
