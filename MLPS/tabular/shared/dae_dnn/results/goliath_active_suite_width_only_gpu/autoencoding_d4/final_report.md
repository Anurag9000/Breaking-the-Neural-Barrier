# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/autoencoding_d4`
- Git commit: `1f870b5535e553c56aaeeda5bad603d469cde33c`
- Device: `cpu`
- Tasks completed: `['autoencoding']`

## Task: autoencoding
- Overall winner: `adp` via `ae_width_only` at `0.000664`
- Winner ADP architecture: `[124, 124, 124, 124]`
- Winner STL architecture: `[124, 124, 124, 124]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [124, 124, 124, 124] | 0.000664 | [124, 124, 124, 124] | 0.003197 | adp |
