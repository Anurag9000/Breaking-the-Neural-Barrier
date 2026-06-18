# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/recovery/missing_width_stl_v1/width_only/simulation/repeat_05/d01`
- Git commit: `7a7eec4fba7b5cbec9304163585cc61beea979e4`
- Device: `cuda`
- Tasks completed: `['simulation']`

## Task: simulation
- Overall winner: `adp` via `ae_width_only` at `0.005123`
- Winner ADP architecture: `[56]`
- Winner STL architecture: `[56]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_only | [56] | 0.005123 | [56] | 0.013209 | adp |
