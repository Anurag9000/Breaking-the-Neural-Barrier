# Repository Structure And Guide

This file is the canonical high-level guide for the cleaned repo layout.

## Top-Level Layout

| Root | Purpose |
| --- | --- |
| `MLPS/` | Fully connected model families. The active staged tabular suite lives in `MLPS/tabular/shared/dae_dnn/`. |
| `CONVS/` | Convolutional models, including CNN, AE, DAE, and related conv variants. |
| `TRANSFORMERS/` | Vision, language, sequence, AE, and DAE transformer families. |
| `RECURRENTS/` | RNN, GRU, LSTM, and recurrent AE/DAE families. |
| `Graph/` | Graph-native model families that are not plain FC wrappers over graph inputs. |
| `Diffusion/` | Diffusion-specific code retained as a separate family due to architecture-specific coupling. |
| `utils/` | Shared ADP contracts, plotting, logging, and migration helpers. |
| `docs/` | Process notes and migration checklists. |

## ADP Wiring

There are now three canonical ADP paths:

1. `MLPS/tabular/shared/dae_dnn/run_goliath_staged_width.py`
   - exact staged search used by the active tabular runs
   - disk-backed candidate state, watchdog-friendly resume, monotonic global best, and per-candidate artefacts

2. `utils/adp_contract.py`
   - shared generic ADP contract for non-tabular MLP families
   - now aligned with the staged runner’s width/depth patience defaults and `width_to_depth` stopping behavior

3. `utils/transformer_mlp_adp.py`
   - shared transformer FFN adapter
   - synchronizes FFN width/depth changes across all transformer blocks while leaving attention untouched

## Current Defaults

- Width expansion patience: `10`
- Depth expansion patience: `2`
- Width-stage margin patience: `10`
- Depth additions in staged transformer and MLP width-to-depth flows only stop on non-improving depth attempts once depth patience is exhausted
- Global best loss is monotonic and must never increase

## Results Layout

Primary run artefacts are under:

- `MLPS/tabular/shared/dae_dnn/results/`

Typical contents per run:

- `training_log.txt`
- `training_stats.csv`
- `candidate_state.json`
- `metadata.json`
- `search_state.json`
- `phase_progress.csv`
- `final_report.json`
- `final_report.md`
- generated plots and analysis CSV/JSON files

Checkpoint binaries may still be excluded from normal Git tracking depending on `.gitignore`.

## Active Experiment Docs

- [README.md](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/README.md)
- [MLPS/tabular/shared/dae_dnn/EXPERIMENT_HANDOFF.md](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/tabular/shared/dae_dnn/EXPERIMENT_HANDOFF.md)
- [TRANSFORMERS/TRANSFORMER_MLP_ADP.md](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/TRANSFORMERS/TRANSFORMER_MLP_ADP.md)

## Cleanup Notes

- Legacy top-level architecture folders that were migrated into consolidated roots should remain deleted.
- Generated inventory dumps and stale scaffold lists should not be the source of truth; the canonical source is the live tree plus the docs listed above.
