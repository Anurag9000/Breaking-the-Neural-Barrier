# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/recovery/missing_width_stl_v1/width_only/prediction/repeat_01/d01`
- Git commit: `f5cc76d508790208e390cb6ea77dd9da007411fd`
- Device: `cpu`
- Tasks completed: `['prediction']`

## Task: prediction
- Overall winner: `stl` via `stl_from_ae_width_only` at `0.313900`
- Winner ADP architecture: `[28]`
- Winner STL architecture: `[28]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [28] | 0.314910 | [28] | 0.313900 | stl |
