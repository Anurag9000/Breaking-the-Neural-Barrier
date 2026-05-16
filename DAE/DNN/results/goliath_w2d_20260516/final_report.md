# DAE/DNN Goliath Final Report

- Run root: `DAE/DNN/results/goliath_w2d_20260516`
- Git commit: `ae063cf34603af76bd877c6937c4090762c0b8c5`
- Device: `cuda`
- Tasks completed: `['prediction', 'representation']`

## Task: prediction
- Overall winner: `adp` via `ae_width_to_depth` at `0.000272`
- Winner ADP architecture: `in=8 hidden=[255, 255] out=8 bn=True`
- Winner STL architecture: `[255, 255]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_to_depth | in=8 hidden=[255, 255] out=8 bn=True | 0.000272 | [255, 255] | 0.332613 | adp |

## Task: representation
- Overall winner: `adp` via `ae_width_to_depth` at `0.000098`
- Winner ADP architecture: `in=54 hidden=[466, 466] out=54 bn=True`
- Winner STL architecture: `[466, 466]`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
| ae_width_to_depth | in=54 hidden=[466, 466] out=54 bn=True | 0.000098 | [466, 466] | 0.227346 | adp |

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

## Task: misc
- Overall winner: `n/a` via `n/a` at `n/a`

| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |
|---|---|---:|---|---:|---|
