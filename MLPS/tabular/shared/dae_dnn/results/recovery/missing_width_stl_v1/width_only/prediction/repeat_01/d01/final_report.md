# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/recovery/missing_width_stl_v1/width_only/prediction/repeat_01/d01`
- Git commit: `7a7eec4fba7b5cbec9304163585cc61beea979e4`
- Device: `cuda`
- Tasks completed: `['prediction']`

## Task: prediction
- Overall winner: `adp` via `ae_width_only` at `0.287792`
- Winner ADP architecture: `[147]`
- Winner STL architecture: `[147]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [147] | 0.287792 | [147] | 0.303297 | adp |
