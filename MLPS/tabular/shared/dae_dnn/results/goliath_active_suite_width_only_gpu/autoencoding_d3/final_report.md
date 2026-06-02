# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/autoencoding_d3`
- Git commit: `f7e455174b8ed5a353ff3233c9891784c3040de5`
- Device: `cpu`
- Tasks completed: `['autoencoding']`

## Task: autoencoding
- Overall winner: `adp` via `ae_width_only` at `0.000823`
- Winner ADP architecture: `[111, 111, 110]`
- Winner STL architecture: `[111, 111, 110]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [111, 111, 110] | 0.000823 | [111, 111, 110] | 0.003419 | adp |
