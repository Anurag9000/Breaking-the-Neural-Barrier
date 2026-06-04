# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/representation_d4`
- Git commit: `a84f4a61c4e2c60a456de13bc443a2243f05822d`
- Device: `cuda`
- Tasks completed: `['representation']`

## Task: representation
- Overall winner: `adp` via `ae_width_only` at `0.086267`
- Winner ADP architecture: `[127, 126, 126, 126]`
- Winner STL architecture: `[127, 126, 126, 126]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [127, 126, 126, 126] | 0.086267 | [127, 126, 126, 126] | 0.208771 | adp |
