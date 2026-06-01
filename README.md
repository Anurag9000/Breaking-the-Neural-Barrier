# Breaking-the-Neural-Barrier

Canonical repo index for layout, ADP wiring, and experiment entry points.

## Architecture Roots

- `MLPS/`: fully connected families, including the active tabular staged suite under `MLPS/tabular/shared/dae_dnn/`
- `CONVS/`: convolutional CNN, AE, DAE, and related vision families
- `TRANSFORMERS/`: text, vision, sequence, AE, and DAE transformer families
- `RECURRENTS/`: LSTM, GRU, RNN, and recurrent AE/DAE families
- `Graph/` and `MLPS/graph/`: graph-native and graph-input model families
- `Diffusion/`: diffusion-specific implementations kept as a separate dependency-coupled family
- `utils/`: shared ADP helpers, plotting, logging, and wrapper contracts

## Canonical ADP Implementations

- MLP staged search used by the active production runs:
  [run_goliath_staged_width.py](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/tabular/shared/dae_dnn/run_goliath_staged_width.py)
- Shared generic ADP contract used by non-tabular MLP wrappers:
  [adp_contract.py](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/utils/adp_contract.py)
- Shared transformer FFN ADP adapter:
  [transformer_mlp_adp.py](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/utils/transformer_mlp_adp.py)

## Current Experiment Docs

- Repo layout and cleanup notes:
  [REPO_STRUCTURE_AND_GUIDE.md](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/REPO_STRUCTURE_AND_GUIDE.md)
- Tabular experiment handoff and run procedure:
  [EXPERIMENT_HANDOFF.md](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/tabular/shared/dae_dnn/EXPERIMENT_HANDOFF.md)
- Transformer FFN ADP contract:
  [TRANSFORMER_MLP_ADP.md](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/TRANSFORMERS/TRANSFORMER_MLP_ADP.md)

## Results

Tabular run outputs, logs, JSON summaries, CSV metrics, plots, and watchdog state are stored under:

- `MLPS/tabular/shared/dae_dnn/results/`

Model checkpoint binaries remain separate from the lightweight metadata/log artefacts when Git ignore rules exclude `*.pt` and `*.ckpt`.
