# Repository Guidelines

## Project Structure & Module Organization
This repository is organized by backbone architecture first, then source family and training paradigm. Use `MLPS/` for fully connected models, `CONVS/` for convolutional families, `TRANSFORMERS/` for attention-based families, `RECURRENTS/` for LSTM/GRU/RNN families, and `Graph/` for graph-native models. `Diffusion/` remains separate because it contains a tightly coupled mix of U-Net, DiT, token, and hybrid diffusion implementations. Shared helpers live in `utils/`. Generated outputs and experiment traces are written to `logs/` or per-run results folders.

ADP variants follow the naming pattern `*_adp_width_to_depth.py`, while base models usually keep a shorter name like `ae_plain.py` or `run_resnet_stl.py`. Keep new files in the matching architecture and source-family folder so the original model and its ADP counterpart stay side by side.

## Build, Test, and Development Commands
There is no project-wide build step. Run the relevant Python entry point directly:

- `python test_gpu.py` - quick CUDA sanity check.
- `python MLPS/tabular/shared/dae_dnn/run_task.py --task classification --mode stl --hidden 50 50 --data-dir ./data --results-dir MLPS/tabular/shared/dae_dnn/results` - run one baseline task.
- `python MLPS/tabular/shared/dae_dnn/run_task.py --task classification --mode adp --adp-mode width_to_depth --hidden 50 50` - run one ADP search.
- `python MLPS/tabular/shared/dae_dnn/run_all.py --data-dir ./data --results-dir MLPS/tabular/shared/dae_dnn/results --hidden 50 50` - run the full tabular MLP task suite.
- `python CONVS/CNN/ADP_ResNet/run_resnet_stl.py` - train the CNN STL baseline with its default arguments.

## Coding Style & Naming Conventions
Use standard Python style: 4-space indentation, `snake_case` for functions and files, and `CamelCase` for classes/dataclasses. Keep model code explicit and readable; avoid clever abstractions that obscure architecture changes. Preserve the original model behavior when editing ADP wrappers, and keep CLI flags aligned with the existing runner conventions (`--adp-mode`, `--max-epochs`, `--results-dir`).

There is no repo-wide formatter or linter config checked in, so match the surrounding style in each folder.

## Testing Guidelines
There is no centralized automated test suite. Validate changes by running the narrowest affected runner plus `test_gpu.py` when CUDA behavior matters. Prefer smoke tests that finish quickly and confirm the training loop, logging, and output files still work. If you add tests, use `test_*.py` naming so they are easy to discover.

## Commit & Pull Request Guidelines
Recent commits are short, imperative, and often scoped to the affected model family, for example: `Fix ADP Wrappers for RNN, CNN (VGG), and Transformers`. Follow that pattern: state what changed, where, and keep the subject line concise.

Pull requests should include:
- a brief summary of the model or runner changed,
- the exact command(s) used to verify it,
- links to related issues or notes,
- screenshots or log snippets only when results or plots changed.

## Data & Artifacts
Do not commit generated datasets, checkpoints, or large experiment artifacts. The `.gitignore` already excludes `data/`, `datasets/`, `*.pt`, `*.pth`, `*.ckpt`, and similar outputs.
