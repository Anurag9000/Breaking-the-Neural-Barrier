# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/representation_d2`
- Git commit: `a84f4a61c4e2c60a456de13bc443a2243f05822d`
- Device: `cuda`
- Tasks completed: `['representation']`

## Task: representation
- Overall winner: `adp` via `ae_width_only` at `0.111073`
- Winner ADP architecture: `[246, 245]`
- Winner STL architecture: `[246, 245]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [246, 245] | 0.111073 | [246, 245] | 0.241843 | adp |
