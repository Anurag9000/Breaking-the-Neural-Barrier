#!/usr/bin/env bash
set -euo pipefail

# Massive End-to-End Orchestrator for Text Transformers (Causal LMs)
# Enforces pressure-aware scheduling, No-Swap guarantees, and executes the complete param sweep.

cd "$(dirname "$0")/../.."

# 1. Apply robust pressure-aware and no-swap OS limits natively
source MLPS/tabular/shared/dae_dnn/runtime_tuning.sh
tabular_runtime_bootstrap

echo "============================================================"
echo " [TEXT TRANSFORMER] MASSIVE EXPERIMENT SUITE INITIATED "
echo "============================================================"

# Environment variables to optimize memory and compute (FlashAttention/Chinchilla principles)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export OMP_NUM_THREADS=$TABULAR_CPU_THREADS

TEXT_RUNNER="TRANSFORMERS/Transformer/Supervised/Runs/run_causal_transformer.py"
TEXT_ADP_MODEL="TRANSFORMERS/Transformer/Supervised/Models/model_causal_transformer_adp_width_to_depth.py"

# Phase 1: Vanilla Ablation (Scaling from small width/depth to Band 10 equivalent)
echo ""
echo ">>> Phase 1: Vanilla Ablation (Param Bands 1-10)"
# Note: Expanding context lengths or vocabulary would blow up VRAM. We scale depth and d_model.
for depth in 1 2 4 8 12; do
    for width in 64 128 256 512 1024 2048; do
        ff=$((width * 4))
        nhead=$((width / 64))
        if [ $nhead -lt 1 ]; then nhead=1; fi
        
        echo "--> Vanilla Ablation: Depth=${depth}, Width=${width}, FF=${ff}, Heads=${nhead}"
        # We wrap in '|| true' so that if a massive configuration OOMs, the suite continues
        python "$TEXT_RUNNER" --layers "$depth" --d_model "$width" --ff "$ff" --nhead "$nhead" --epochs 10 --batch_size 16 || echo "Config ($depth, $width) failed/OOMed. Continuing..."
    done
done

# Phase 2: ADP Width-Only Suite (Depths 1 to 5)
echo ""
echo ">>> Phase 2: ADP Width-Only Suite (Depths 1 to 5)"
for depth in 1 2 3 4 5; do
    echo "--> ADP Width-Only Search: Initial Depth=${depth}"
    python "$TEXT_ADP_MODEL" --adp-mode width_only --depth "$depth" --width 64 --max-epochs 10 || echo "ADP Depth $depth failed."
done

# Phase 3: ADP Width-to-Depth Suite
echo ""
echo ">>> Phase 3: ADP Width-to-Depth (W2D) Suite"
echo "--> Starting dynamic w2d search from minimal seed (Depth=1, Width=64)"
python "$TEXT_ADP_MODEL" --adp-mode width_to_depth --depth 1 --width 64 --max-epochs 10 || echo "ADP W2D failed."

echo "============================================================"
echo " [TEXT TRANSFORMER] MASSIVE EXPERIMENT SUITE COMPLETED "
echo "============================================================"
