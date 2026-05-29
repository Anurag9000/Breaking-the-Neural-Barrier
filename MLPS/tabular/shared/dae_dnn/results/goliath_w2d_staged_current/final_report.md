# DAE/DNN Goliath Final Report

- Run root: `MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current`
- Git commit: `7fb2f9b4091463e4f24bc808eca55fed48513922`
- Device: `cpu`
- Tasks completed: `['representation', 'autoencoding', 'generation']`

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
- Overall winner: `adp` via `ae_width_to_depth` at `0.001482`
- Winner ADP architecture: `[81]`
- Winner STL architecture: `[81]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_alt_depth | [89, 89, 89, 89, 88, 88] | 0.001903 | [89, 89, 89, 89, 88, 88] | 0.013283 | adp |
| ae_alt_width | [83] | 0.001727 | [83] | 0.003410 | adp |
| ae_width_to_depth | [81] | 0.001482 | [81] | 0.003484 | adp |
| ae_depth_to_width | [101, 101, 101, 101, 101, 100, 100, 100, 100, 100] | 0.001521 | [101, 101, 101, 101, 101, 100, 100, 100, 100, 100] | 0.023918 | adp |

## Task: generation
- Overall winner: `adp` via `ae_alt_width` at `0.000248`
- Winner ADP architecture: `[59]`
- Winner STL architecture: `[59]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_alt_depth | [96, 96, 96, 95, 95, 95] | 0.000699 | [96, 96, 96, 95, 95, 95] | 1.008456 | adp |
| ae_alt_width | [59] | 0.000248 | [59] | 1.008655 | adp |
| ae_width_to_depth | [56] | 0.000308 | [56] | 1.008609 | adp |
| ae_depth_to_width | [140, 140, 140, 140, 140, 139, 139, 139, 139, 139] | 0.001788 | [140, 140, 140, 140, 140, 139, 139, 139, 139, 139] | 1.008333 | adp |

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
