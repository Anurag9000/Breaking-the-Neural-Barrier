# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/representation_d3`
- Git commit: `a84f4a61c4e2c60a456de13bc443a2243f05822d`
- Device: `cuda`
- Tasks completed: `['representation']`

## Task: representation
- Overall winner: `adp` via `ae_width_only` at `0.094178`
- Winner ADP architecture: `[151, 151, 151]`
- Winner STL architecture: `[151, 151, 151]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [151, 151, 151] | 0.094178 | [151, 151, 151] | 0.189461 | adp |
