# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/recovery/missing_width_stl_v1/width_only/autoencoding/repeat_02/d01`
- Git commit: `e81c9a7a20f6137408639f3213bb3a5aaefc218d`
- Device: `cuda`
- Tasks completed: `['autoencoding']`

## Task: autoencoding
- Overall winner: `adp` via `ae_width_only` at `0.000539`
- Winner ADP architecture: `[74]`
- Winner STL architecture: `[74]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [74] | 0.000539 | [74] | 0.001155 | adp |
