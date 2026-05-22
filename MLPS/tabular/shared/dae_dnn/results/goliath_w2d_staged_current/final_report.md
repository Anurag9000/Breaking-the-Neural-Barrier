# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current`
- Git commit: `338cf805dc13f0d8d9943124973b96f851dafa10`
- Device: `cuda`
- Tasks completed: `['representation']`

## Task: representation
- Overall winner: `adp` via `ae_width_to_depth` at `0.086187`
- Winner ADP architecture: `[161, 160, 160, 160]`
- Winner STL architecture: `[161, 160, 160, 160]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_alt_depth | [105, 105, 105, 104, 104, 104] | 0.088987 | [105, 105, 105, 104, 104, 104] | 0.186349 | adp |
| ae_alt_width | [227, 227, 227, 227, 227, 227, 227, 227, 226, 226] | 0.090317 | [227, 227, 227, 227, 227, 227, 227, 227, 226, 226] | 0.145708 | adp |
| ae_width_to_depth | [161, 160, 160, 160] | 0.086187 | [161, 160, 160, 160] | 0.200954 | adp |
| ae_depth_to_width | [93, 93, 93, 93, 93, 93, 93, 92, 92, 92] | 0.087505 | [93, 93, 93, 93, 93, 93, 93, 92, 92, 92] | 0.164796 | adp |

## Task: autoencoding
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|

## Task: generation
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|

## Task: denoising
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|

## Task: anomaly
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|

## Task: inverse
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|

## Task: control
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|

## Task: clustering
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|

## Task: compression
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|

## Task: ranking
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|

## Task: multimodal
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|

## Task: selfsupervised
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|

## Task: simulation
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
