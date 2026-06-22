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

for run_idx in {1..5}; do
    echo ""
    echo "############################################################"
    echo ">>> EXPERIMENT REPEAT: $run_idx OF 5"
    echo "############################################################"

    # Phase 1: Vanilla Ablation (Scaling from small width/depth to Band 10 equivalent)
    echo ""
    echo ">>> Phase 1: Vanilla Ablation (Param Bands 1-10)"
    
    # Generate dynamic (depth, width) pairs targeting param bands 1 to 10
    # Using 3 samples per band (e.g. 1e4, 5e4, 1e5...)
    GRID=$($Python utils/generate_ablation_grid.py --arch text --min-band 1 --max-band 10 --samples 3 --depths 1,2,4,8,12)
    
    echo "$GRID" | while read -r depth width; do
        ff=$((width * 4))
        nhead=$((width / 64))
        if [ "$nhead" -lt 1 ]; then nhead=1; fi
        
        echo "--> Vanilla Ablation: Depth=$depth, Width=$width, FF=$ff, Heads=$nhead"
        $Python utils/pressure_aware_wrapper.py $Python "$TEXT_RUNNER" --layers "$depth" --d_model "$width" --ff "$ff" --nhead "$nhead" --epochs 10 --batch_size 16 || echo "OOM or Failed. Continuing..."
    done

    # Phase 2: ADP Width-Only Suite (Depths 1 to 5)
    echo ""
    echo ">>> Phase 2: ADP Width-Only Suite (Depths 1 to 5)"
    for depth in 1 2 3 4 5; do
        echo "--> ADP Width-Only Search: Initial Depth=${depth}"
        python utils/pressure_aware_wrapper.py python "$TEXT_ADP_MODEL" --adp-mode width_only --depth "$depth" --width 64 --max-epochs 10 || echo "ADP Depth $depth failed."
    done

    # Phase 3: ADP Width-to-Depth Suite
    echo ""
    echo ">>> Phase 3: ADP Width-to-Depth (W2D) Suite"
    echo "--> Starting dynamic w2d search from minimal seed (Depth=1, Width=64)"
    python utils/pressure_aware_wrapper.py python "$TEXT_ADP_MODEL" --adp-mode width_to_depth --depth 1 --width 64 --max-epochs 10 || echo "ADP W2D failed."
done

echo "============================================================"
echo " [TEXT TRANSFORMER] MASSIVE EXPERIMENT SUITE COMPLETED "
echo "============================================================"
