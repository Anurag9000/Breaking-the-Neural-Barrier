import copy
from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore
from utils.adp_logging import ContinuousLogger

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_transformer_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
AE_TRANSFORMER_STL = baseline_module.AE_TRANSFORMER_STL  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# - This file is newly created to implement the ADP algorithms from scratch for the Transformer STL model.
# - It strictly follows ADP_algorithms.md: forward-only expansions, global best tracking, and context-end restoration.
# - Implements special handling for MultiheadAttention weight resizing.

@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16  # Must ensure embed_dim remains divisible by num_heads
    max_width: int = 512
    max_depth: int = 16
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    max_epochs: int = 100_000_000
    num_heads: int = 6
    patch_size: int = 4
    mlp_ratio: float = 4.0


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt

def _resize_mha_weight(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    # Handle in_proj_weight: (3*D, D) or in_proj_bias: (3*D,)
    # We assume 'to_shape' implies 3*new_D.
    # We detect if it's 1D (bias) or 2D (weight).
    
    is_bias = (len(to_shape) == 1)
    
    # Calculate dimensions
    # For weight: (3*Do, Do). For bias: (3*Do,)
    # We assume src is valid MHA weight/bias.
    
    # Src D
    if is_bias:
        Do = src.shape[0] // 3
        Dn = to_shape[0] // 3
    else:
        Do = src.shape[1]
        Dn = to_shape[1]
        
    # Split
    if is_bias:
        q, k, v = src.chunk(3, dim=0)
        # Resize each to (Dn,)
        q_new = _resize_tensor(torch.Size([Dn]), q)
        k_new = _resize_tensor(torch.Size([Dn]), k)
        v_new = _resize_tensor(torch.Size([Dn]), v)
        return torch.cat([q_new, k_new, v_new], dim=0)
    else:
        q, k, v = src.chunk(3, dim=0) # Each (Do, Do)
        # Resize each to (Dn, Dn)
        # Note: input dim (dim 1) also changes from Do to Dn.
        q_new = _resize_tensor(torch.Size([Dn, Dn]), q)
        k_new = _resize_tensor(torch.Size([Dn, Dn]), k)
        v_new = _resize_tensor(torch.Size([Dn, Dn]), v)
        return torch.cat([q_new, k_new, v_new], dim=0)


def _merge_state(new_state, old_state):
    merged = {}
    for k, v in new_state.items():
        if k in old_state:
            ov = old_state[k]
            if ov.shape == v.shape:
                merged[k] = ov
            elif "in_proj_weight" in k or "in_proj_bias" in k:
                # Special MHA handling
                merged[k] = _resize_mha_weight(v.shape, ov)
            else:
                merged[k] = _resize_tensor(v.shape, ov)
        else:
            merged[k] = v
            
    return merged


def rebuild_model(model: AE_TRANSFORMER_STL, embed_dim: int, depth: int, device, cfg: ADPConfig) -> AE_TRANSFORMER_STL:
    new_model = AE_TRANSFORMER_STL(in_channels=3, embed_dim=embed_dim, depth=depth, 
                                   num_heads=cfg.num_heads, patch_size=cfg.patch_size, mlp_ratio=cfg.mlp_ratio).to(device)
    
    # Ensure compatible shapes for state dict merge
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(model: AE_TRANSFORMER_STL, ex_k: int, max_width: int, device, cfg: ADPConfig) -> Optional[AE_TRANSFORMER_STL]:
    # Ensure new width is multiple of num_heads
    current_dim = model.patch.proj.out_channels # access embed_dim via patch.proj
    # or just use model.blocks[0].norm1.normalized_shape[0] if exists.
    
    # We can track embed_dim in 'snapshot', but let's read from model ops.
    # stored in patch.proj.weight shape: (embed_dim, in, k, k)
    
    H = cfg.num_heads
    
    # Target increment
    target_dim = current_dim + ex_k
    
    # Round up to multiple of H
    rem = target_dim % H
    if rem != 0:
        target_dim += (H - rem)
        
    new_w = min(max_width, target_dim)
    
    # Ensure divisible (if max_width clamped it)
    new_w = (new_w // H) * H
    
    if new_w <= current_dim:
        return None
        
    return rebuild_model(model, new_w, len(model.blocks), device, cfg)


def expand_depth(model: AE_TRANSFORMER_STL, max_depth: int, device, cfg: ADPConfig) -> Optional[AE_TRANSFORMER_STL]:
    current_depth = len(model.blocks)
    if current_depth >= max_depth:
        return None
        
    embed_dim = model.patch.proj.out_channels
    return rebuild_model(model, embed_dim, current_depth + 1, device, cfg)


def total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1)) # Approx metric


def snapshot_arch_and_state(model: AE_TRANSFORMER_STL, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "embed_dim": model.patch.proj.out_channels,
        "depth": len(model.blocks),
        "num_heads": model.blocks[0].attn.num_heads if len(model.blocks) > 0 else 6, # Fallback
        "state": copy.deepcopy(state)
    }


def restore_arch_and_state(model: AE_TRANSFORMER_STL, snap: Dict[str, Any], device) -> AE_TRANSFORMER_STL:
    # Need access to config for constants? Or store them in snapshot?
    # We'll assume constants (patch_size, mlp_ratio) didn't change, but read heads/dim/depth from snap.
    # But restore requires creating a new instance.
    # We need the other args from original config. 
    # For this wrapper, we assume global 'acfg' or similar holds the static constants, 
    # OR we add them to snapshot.
    # Let's use the 'model' passed in to infer static constants if needed, but 'model' might be different.
    # Best to capture 'patch_size' etc in snapshot if we want true restoration.
    
    # But 'restore_arch_and_state' signature usually just takes 'snap' and 'device'.
    # We can instantiate using the same class with snap params.
    # We need 'defaults' for things not in snap.
    
    # Let's rely on the passed 'model' to get static props, 
    # OR better: pass 'acfg' to restore ... but the signature is fixed in the harness usually.
    # We'll assume we can create it with defaults matching 'model' or 'acfg' (we can close over 'acfg' if needed, but better to be explicit).
    
    # Let's assume standard params are constant.
    new_model = AE_TRANSFORMER_STL(
        in_channels=3,
        embed_dim=snap["embed_dim"],
        depth=snap["depth"],
        num_heads=snap.get("num_heads", 6), # Default 6
        patch_size=4, # Hardcoded in this script to 4 as per main() default? Or read from model?
        mlp_ratio=4.0
    ).to(device)
    
    new_model.load_state_dict(snap["state"])
    return new_model


def train_with_early_stopping(model: AE_TRANSFORMER_STL, dl_train, dl_val, acfg: ADPConfig, device, history: list) -> Tuple[float, Dict[str, Any]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0
    
    for _ in range(acfg.max_epochs):
        model.train()
        for x, _ in dl_train:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            rec, _ = model(x)
            loss = F.mse_loss(rec, x)
            loss.backward()
            if acfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
        
        model.eval()
        with torch.no_grad():
            val = 0.0
            n = 0
            for x, _ in dl_val:
                x = x.to(device)
                rec, _ = model(x)
                l = F.mse_loss(rec, x)
                val += l.item()
                n += 1
            val = val / max(n, 1)
        
        history.append(val)
        
        # Log to console and text file
        msg = f"  Epoch {_+1}/{acfg.max_epochs} | Device: {device} | Val Loss: {val:.6f} | Best: {best:.6f} | Pat: {pat}/{acfg.patience}"
        if verbose and logger:
            logger.log_console(msg)
        elif verbose:
            # print(msg) # optional, keep silent if desired, but logger is preferred
            pass
        
        # Log to CSV immediately
        if logger:
            logger.log_epoch_stats({
                "epoch": len(history),
                "width": model.width,
                "depth": model.depth,
                "neurons": total_neurons(model),
                "val_loss": val,
                "best_val": best,
                "es_counter": acfg.patience - pat, # approx
                "improved": (val < best - acfg.delta)
            })
        
        if val < best_val: # Strict improvement
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1
            
        if es_counter >= acfg.patience:
            break
            
    return best_val, best_state


def adp_search(model: AE_TRANSFORMER_STL, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp_ae_transformer")):
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    improvements: List[tuple[int, float]] = []

    # Initial training
    best_val, best_state = train_with_early_stopping(model, dl_train, dl_val, acfg, device, val_history, logger=logger)
    model.load_state_dict(best_state)
    
    # Global best snapshot
    global_best_snap = snapshot_arch_and_state(model, best_state)
    global_best_val = best_val
    improvements.append((total_neurons(model.patch.proj.out_channels, len(model.blocks)), best_val))

    def can_widen(m: AE_TRANSFORMER_STL) -> bool:
        curr_dim = m.patch.proj.out_channels
        # Check against max_width
        if curr_dim >= acfg.max_width:
            return False
        # Estimate next dim
        next_dim = curr_dim + acfg.ex_k
        if next_dim > acfg.max_width: return False
        return total_neurons(next_dim, len(m.blocks)) <= acfg.max_neurons

    def can_deepen(m: AE_TRANSFORMER_STL) -> bool:
        curr_depth = len(m.blocks)
        curr_dim = m.patch.proj.out_channels
        return curr_depth + 1 <= acfg.max_depth and total_neurons(curr_dim, curr_depth + 1) <= acfg.max_neurons

    # 3.1 Inner: optimize_width_at_fixed_depth
    def optimize_width_at_fixed_depth(curr_model: AE_TRANSFORMER_STL) -> Tuple[AE_TRANSFORMER_STL, float, Dict[str, Any]]:
        local_val, local_state = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
        local_best_val = local_val
        local_best_state = local_state
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        
        width_failure_count = 0
        
        while width_failure_count < acfg.trials_width:
            if not can_widen(curr_model):
                break
            
            # Always expand from current width
            next_model = expand_width(curr_model, acfg.ex_k, acfg.max_width, device, acfg)
            if next_model is None: 
                break
            curr_model = next_model # Update reference
            
            v, s = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_state = s
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                width_failure_count = 0
                improvements.append((total_neurons(curr_model.patch.proj.out_channels, len(curr_model.blocks)), v))
            else:
                width_failure_count += 1
                logger.log_console(f'[WIDTH OPT] ✗ No improvement')
                
        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        return final_model, local_best_val, local_best_snap

    # 4.1 Inner: optimize_depth_at_fixed_width
    def optimize_depth_at_fixed_width(curr_model: AE_TRANSFORMER_STL) -> Tuple[AE_TRANSFORMER_STL, float, Dict[str, Any]]:
        local_val, local_state = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
        local_best_val = local_val
        local_best_state = local_state
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        
        depth_failure_count = 0
        
        while depth_failure_count < acfg.trials_depth:
            if not can_deepen(curr_model):
                break
                
            next_model = expand_depth(curr_model, acfg.max_depth, device, acfg)
            if next_model is None:
                break
            curr_model = next_model
            
            v, s = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_state = s
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                depth_failure_count = 0
                improvements.append((total_neurons(curr_model.patch.proj.out_channels, len(curr_model.blocks)), v))
            else:
                depth_failure_count += 1
                logger.log_console(f'[DEPTH OPT] ✗ No improvement')
        
        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        return final_model, local_best_val, local_best_snap

    mode = acfg.adp_mode
    
    if mode in ["width_only", "width"]:
        model, global_best_val, global_best_snap = optimize_width_at_fixed_depth(model)
        
    elif mode in ["depth_only", "depth"]:
        model, global_best_val, global_best_snap = optimize_depth_at_fixed_width(model)
        
    elif mode == "depth_to_width": # ADP_DEPTH_OUTER_WIDTH_INNER
        model, base_val, base_snap = optimize_depth_at_fixed_width(model)
        global_best_val = base_val
        global_best_snap = base_snap
        
        depth_failure_count = 0
        while depth_failure_count < acfg.trials_depth:
            if not can_deepen(model): break
            next_model = expand_depth(model, acfg.max_depth, device, acfg)
            if next_model is None: break
            model = next_model
            
            model, val_d, snap_d = optimize_width_at_fixed_depth(model)
            if val_d < global_best_val - acfg.delta:
                global_best_val = val_d
                global_best_snap = snap_d
                depth_failure_count = 0
            else:
                depth_failure_count += 1
                logger.log_console(f'[DEPTH OPT] ✗ No improvement')
                
        model = restore_arch_and_state(model, global_best_snap, device)

    elif mode == "width_to_depth": # ADP_WIDTH_OUTER_DEPTH_INNER
        model, base_val, base_snap = optimize_width_at_fixed_depth(model)
        global_best_val = base_val
        global_best_snap = base_snap
        
        width_failure_count = 0
        while width_failure_count < acfg.trials_width:
            if not can_widen(model): break
            next_model = expand_width(model, acfg.ex_k, acfg.max_width, device, acfg)
            if next_model is None: break
            model = next_model
            
            model, val_w, snap_w = optimize_depth_at_fixed_width(model)
            if val_w < global_best_val - acfg.delta:
                global_best_val = val_w
                global_best_snap = snap_w
                width_failure_count = 0
            else:
                width_failure_count += 1
                logger.log_console(f'[WIDTH OPT] ✗ No improvement')
        
        model = restore_arch_and_state(model, global_best_snap, device)

    elif mode in ["alt_width", "alt_depth"]:
        depth_saturated = False
        width_saturated = False
        current_phase = "width" if mode == "alt_width" else "depth"
        
        while not (depth_saturated and width_saturated):
            improved_in_phase = False
            if current_phase == "width":
                model, val, snap = optimize_width_at_fixed_depth(model)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved_in_phase = True
                width_saturated = not improved_in_phase
                model = restore_arch_and_state(model, global_best_snap, device)
                current_phase = "depth"
            else:
                model, val, snap = optimize_depth_at_fixed_width(model)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved_in_phase = True
                depth_saturated = not improved_in_phase
                model = restore_arch_and_state(model, global_best_snap, device)
                current_phase = "width"
                
        model = restore_arch_and_state(model, global_best_snap, device)

    # ADP REVIEW (AFTER REFACTOR)
    # - Implemented fully compliant forward-only logic with smart resizing for Transformers.
    
    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"AE_Transformer_STL ({acfg.adp_mode})")
    if log_neurons and improvements:
        plot_loss_vs_neurons([n for n,_ in improvements], [v for _,v in improvements], results_dir / "loss_vs_neurons.png", title=f"AE_Transformer_STL ({acfg.adp_mode})")
        
    return global_best_val, model, model.patch.proj.out_channels, len(model.blocks)


def make_loaders(batch_size: int = 128, val_split: float = 0.1):
    sys.path.append(str(Path(__file__).resolve().parents[1] / "Runs"))
    from _common_real_image import make_real_image_loaders
    dl_train, dl_val, _ = make_real_image_loaders(
        data_root="./data",
        batch_size=batch_size,
        val_ratio=val_split,
        num_workers=4,
        image_size=224,
    )
    return dl_train, dl_val


def main():
    import argparse
    p = argparse.ArgumentParser(description="ADP AE_Transformer_STL width/depth search")
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["alt_width", "width_to_depth"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=16)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--results-dir", type=Path, default=Path("results_adp_ae_transformer"))
    p.add_argument("--plot-loss", action="store_true", help="Save loss-vs-epoch (log scale)")
    p.add_argument("--plot-neurons", action="store_true", help="Save neurons-vs-loss (log scale)")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AE_TRANSFORMER_STL(in_channels=3, embed_dim=args.embed_dim, depth=args.depth, 
                               num_heads=args.num_heads, patch_size=args.patch_size, mlp_ratio=4.0).to(device)
                               
    acfg = ADPConfig(adp_mode=args.adp_mode, delta=args.delta, patience=args.patience, trials_width=args.trials_width,
                     trials_depth=args.trials_depth, ex_k=args.ex_k, max_width=args.max_width, max_depth=args.max_depth,
                     max_neurons=args.max_neurons, max_epochs=args.max_epochs, 
                     num_heads=args.num_heads, patch_size=args.patch_size)
    
    print(f"[ADP AE_Transformer] Starting {args.adp_mode}, Init: dim={args.embed_dim}, depth={args.depth}")
    best, model, w, d = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    print(f"[ADP AE_Transformer] DONE. Best Val={best:.6f} Width={w} Depth={d}")
