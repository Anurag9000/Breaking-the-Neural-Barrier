"""
ADP wrapper for curriculum DAE.

Implements Curriculum Learning (Bengio, 2009) applied to DAE noise levels.
Strategy: Easy-to-Hard. Start with low noise, linearly increase to target `noise_std`.
"""

import copy
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons

# Reuse the model definition, but we need our own training logic
from .dae_gaussian_conv_sup_stl import SupDAEGaussianConv, sup_dae_total_neurons
from .dae_gaussian_conv_sup_stl_adp_width_to_depth import (
    add_gaussian_noise,
    _merge_state,
    rebuild_model,
    expand_width,
    expand_depth,
    snapshot_arch_and_state,
    restore_arch_and_state,
    make_loaders, # Reuse loader logic
)

@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 512
    max_depth: int = 16
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 300
    
    # Curriculum specific
    target_noise_std: float = 0.2
    # Fraction of max_epochs to reach target noise. e.g. 0.5 means first 50% epochs are warmup.
    curriculum_frac: float = 0.5 
    lambda_recon: float = 1.0


def get_current_noise(epoch: int, max_epochs: int, cfg: ADPConfig) -> float:
    """Computes noise std for the current epoch (1-indexed)."""
    warmup_epochs = int(max_epochs * cfg.curriculum_frac)
    if warmup_epochs < 1: return cfg.target_noise_std
    
    if epoch >= warmup_epochs:
        return cfg.target_noise_std
    else:
        # Linear ramp: 0 -> target
        return cfg.target_noise_std * (epoch / warmup_epochs)

def train_with_early_stopping(
    model: SupDAEGaussianConv,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    history: List[float],
    logger: Optional[ContinuousLogger] = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for epoch in range(1, acfg.max_epochs + 1):
        # Update Curriculum
        curr_sigma = get_current_noise(epoch, acfg.max_epochs, acfg)
        
        model.train()
        total, n = 0.0, 0
        for xb, yb in dl_train:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            # Apply dynamic noise
            xb_noisy = add_gaussian_noise(xb, curr_sigma)

            opt.zero_grad(set_to_none=True)
            xb_rec, logits = model(xb_noisy)
            loss_recon = mse(xb_rec, xb)
            loss_cls = ce(logits, yb)
            loss = acfg.lambda_recon * loss_recon + loss_cls
            loss.backward()
            if acfg.grad_clip is not None and acfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()

            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_loss = total / max(n, 1)

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                
                # Validation always at CURRENT curriculum difficulty? 
                # Or Target? Standard CL suggests validating at current difficulty to track progress, 
                # but final performance matters at TARGET difficulty.
                # Ideally, we should validate at TARGET noise to ensure we are learning robust features.
                # However, early in curriculum, loss at target noise might be huge and confusing.
                # Let's use Target Noise for validation to have a consistent metric for Early Stopping.
                
                xb_noisy = add_gaussian_noise(xb, acfg.target_noise_std)
                
                xb_rec, logits = model(xb_noisy)

                loss_recon = mse(xb_rec, xb) / xb.size(0)
                loss_cls = ce(logits, yb) / xb.size(0)
                loss = acfg.lambda_recon * loss_recon + loss_cls
                total += float(loss.item()) * xb.size(0)
                n += xb.size(0)
        val_loss = total / max(n, 1)
        history.append(val_loss)

        improved = val_loss < best_val - acfg.delta
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1

        msg = (
            f"  Epoch {epoch:03d}/{acfg.max_epochs} | Sigma: {curr_sigma:.3f} | "
            f"Train={train_loss:.6f} | Val={val_loss:.6f} | "
            f"Best={best_val:.6f} | ES={es_counter}/{acfg.patience}"
        )
        if logger:
            logger.log_console(msg)
        else:
            print(msg)

        if es_counter >= acfg.patience:
            if logger:
                logger.log_console(f"  Early stopping at epoch {epoch}")
            else:
                print(f"  Early stopping at epoch {epoch}")
            break

    return best_val, best_state

# Re-implement adp_search to use our new train function
def adp_search(
    model: SupDAEGaussianConv, 
    dl_train: DataLoader, 
    dl_val: DataLoader, 
    acfg: ADPConfig, 
    device: torch.device, 
    logger: ContinuousLogger,
    log_loss: bool = False, 
    log_neurons: bool = False,
    results_dir: Path = Path("results_adp")
):
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    
    # Initial Training
    logger.log_console("[INITIAL TRAINING] Starting Curriculum DAE Search...")
    best_val, best_state = train_with_early_stopping(model, dl_train, dl_val, acfg, device, val_history, logger)
    model.load_state_dict(best_state)
    
    global_best_val = best_val
    global_best_snap = snapshot_arch_and_state(model, best_state)
    
    # ... (Rest of ADP logic is standard, but must use our train function)
    # Since I cannot easily import partials without mess, I will copy the standard structure 
    # but call local `train_with_early_stopping`.
    
    # Standard ADP Logic (Copied & Adapted for Context)
    
    def can_widen(m): return m.width + acfg.ex_k <= acfg.max_width
    def can_deepen(m): return m.depth + 1 <= acfg.max_depth
    
    def optimize_width(curr):
        l_val, l_state = train_with_early_stopping(curr, dl_train, dl_val, acfg, device, val_history, logger)
        l_snap = snapshot_arch_and_state(curr, l_state)
        fail = 0
        while fail < acfg.trials_width:
            if not can_widen(curr): break
            nxt = expand_width(curr, curr.num_classes, acfg.ex_k, acfg.max_width, device)
            if nxt is None: break
            curr = nxt # Update curr_model
            v, s = train_with_early_stopping(curr, dl_train, dl_val, acfg, device, val_history, logger)
            if v < l_val - acfg.delta:
                l_val = v; l_snap = snapshot_arch_and_state(curr, s)
                fail = 0
                logger.log_console(f"[WIDTH] Improvement: {v:.4f}")
            else:
                fail += 1
                logger.log_console(f"[WIDTH] No improve. Fail {fail}")
        return restore_arch_and_state(l_snap, device), l_val, l_snap

    def optimize_depth(curr):
        l_val, l_state = train_with_early_stopping(curr, dl_train, dl_val, acfg, device, val_history, logger)
        l_snap = snapshot_arch_and_state(curr, l_state)
        fail = 0
        while fail < acfg.trials_depth:
            if not can_deepen(curr): break
            nxt = expand_depth(curr, curr.num_classes, acfg.max_depth, device)
            if nxt is None: break
            curr = nxt
            v, s = train_with_early_stopping(curr, dl_train, dl_val, acfg, device, val_history, logger)
            if v < l_val - acfg.delta:
                l_val = v; l_snap = snapshot_arch_and_state(curr, s)
                fail = 0
                logger.log_console(f"[DEPTH] Improvement: {v:.4f}")
            else:
                fail += 1
                logger.log_console(f"[DEPTH] No improve. Fail {fail}")
        return restore_arch_and_state(l_snap, device), l_val, l_snap

    # Execution based on mode
    if acfg.adp_mode == "width_to_depth":
        model, bv, bs = optimize_width(model)
        model, bv, bs = optimize_depth(model)
    # ... (Other modes can be added, but minimal required is width_to_depth for now)
    
    return bv, model, model.width, model.depth


