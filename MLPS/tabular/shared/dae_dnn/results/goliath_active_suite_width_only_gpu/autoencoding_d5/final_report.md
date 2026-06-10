# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/autoencoding_d5`
- Git commit: `1f870b5535e553c56aaeeda5bad603d469cde33c`
- Device: `cpu`
- Tasks completed: `['autoencoding']`

## Task: autoencoding
- Overall winner: `adp` via `ae_width_only` at `0.000588`
- Winner ADP architecture: `[127, 126, 126, 126, 126]`
- Winner STL architecture: `[127, 126, 126, 126, 126]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [127, 126, 126, 126, 126] | 0.000588 | [127, 126, 126, 126, 126] | 0.003666 | adp |
