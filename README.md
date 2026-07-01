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

## Transformer Massive Ablation Suites (Vision & Text)

Fully wired end-to-end orchestration scripts are provided to natively execute the entire sweep across Vision (DeiT) and Text (Causal LLM) architectures. These scripts natively support our ADP algorithm on the Transformer MLP blocks, enforce OS-level zero-swap constraints, and orchestrate the full `1 to 10 band` vanilla ablations followed by `width_only` (depths 1-5) and full `width_to_depth` scaling. The entire suite automatically loops **5 times (x5 repeat)** for each configuration to gather robust variance and statistical significance.

**To run the Text Transformer (Causal LM) Suite:**
*   **Linux:** `./TRANSFORMERS/text/run_massive_text_suite.sh`
*   **Windows:** `.\TRANSFORMERS\text\run_massive_text_suite.ps1`

**To run the Vision Transformer (DeiT) Suite:**
*   **Linux:** `./TRANSFORMERS/vision/run_massive_vision_suite.sh`
*   **Windows:** `.\TRANSFORMERS\vision\run_massive_vision_suite.ps1`

## GPU-First Dual-Gate Scheduler

In addition to the standard `--scheduler pressure_aware` logic, all major orchestrator entry points now support `--scheduler gpu_first`. 
This implements a dual-admission gate priority system:
1. **GPU Priority**: The orchestrator will exhaustively launch jobs on the GPU until the VRAM is fully saturated (or a CUDA OOM forces the GPU gate to close).
2. **CPU Fallback**: Once the GPU is confirmed to be full, the orchestrator begins launching jobs on the CPU until the system RAM is saturated.

This ensures zero compute cycles are wasted waiting for the GPU if host RAM is available, while strictly preventing the CPU from "stealing" jobs when the GPU is still cooling down or has available space. 
Counterpart scripts (`*_gpu_first.sh` and `*_gpu_first.ps1`) are provided for the entire 10^1 to 10^10 parameter bands.

## Emergency Operations

If you need to instantly terminate all running MLPS/tabular Python models, generators, or child processes on your system, use the cross-platform emergency kill switch. This will exhaustively scan for and kill all orphaned or running Python processes associated with the pipeline.

**On Linux:**
```bash
./scripts/kill_all_runners.sh
```

**On Windows:**
```powershell
.\scripts\kill_all_runners.ps1
```
