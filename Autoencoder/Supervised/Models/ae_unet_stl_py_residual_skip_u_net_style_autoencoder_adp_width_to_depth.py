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
BASE_PATH = Path(__file__).with_name("ae_unet_stl_py_residual_skip_u_net_style_autoencoder.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
AE_UNET_STL = baseline_module.AE_UNET_STL  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# - This file is newly created to implement the ADP algorithms from scratch for the UNet STL model.
# - It strictly follows ADP_algorithms.md: forward-only expansions, global best tracking, and context-end restoration.

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
    grad_clip: Optional[float] = 1.0
    max_epochs: int = 100_000_000
    pool_after: List[int] = None


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt


def _merge_state(new_state, old_state):
    merged = {}
    for k, v in new_state.items():
        if k in old_state:
            ov = old_state[k]
            merged[k] = ov if ov.shape == v.shape else _resize_tensor(v.shape, ov)
        else:
            merged[k] = v
    return merged


def rebuild_model(model: AE_UNET_STL, width: int, depth: int, device, pool_after: List[int]) -> AE_UNET_STL:
    new_model = AE_UNET_STL(in_channels=model.in_channels, width=width, depth=depth, pool_after=pool_after).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(model: AE_UNET_STL, ex_k: int, max_width: int, device) -> Optional[AE_UNET_STL]:
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return None
    return rebuild_model(model, new_w, model.depth, device, list(model.pool_after))


def expand_depth(model: AE_UNET_STL, max_depth: int, device) -> Optional[AE_UNET_STL]:
    if model.depth >= max_depth:
        return None
    return rebuild_model(model, model.width, model.depth + 1, device, list(model.pool_after))


def total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))


def snapshot_arch_and_state(model: AE_UNET_STL, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "width": model.width,
        "depth": model.depth,
        "pool_after": list(model.pool_after),
        "state": copy.deepcopy(state)
    }


def restore_arch_and_state(model: AE_UNET_STL, snap: Dict[str, Any], device) -> AE_UNET_STL:
    restored = AE_UNET_STL(
        in_channels=model.in_channels,
        width=snap["width"],
        depth=snap["depth"],
        pool_after=list(snap["pool_after"])
    ).to(device)
    restored.load_state_dict(snap["state"])
    return restored


def train_with_early_stopping(model: AE_UNET_STL, dl_train, dl_val, acfg: ADPConfig, device, history: list) -> Tuple[float, Dict[str, Any]]:
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
        
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1
            
        if es_counter >= acfg.patience:
            break
            
    return best_val, best_state


def adp_search(model: AE_UNET_STL, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp_ae_unet")):
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    improvements: List[tuple[int, float]] = []

    # Initial training
    best_val, best_state = train_with_early_stopping(model, dl_train, dl_val, acfg, device, val_history, logger=logger)
    model.load_state_dict(best_state)
    
    global_best_snap = snapshot_arch_and_state(model, best_state)
    global_best_val = best_val
    improvements.append((total_neurons(model.width, model.depth), best_val))

    def can_widen(m: AE_UNET_STL) -> bool:
        new_w = min(acfg.max_width, m.width + acfg.ex_k)
        return new_w > m.width and total_neurons(new_w, m.depth) <= acfg.max_neurons

    def can_deepen(m: AE_UNET_STL) -> bool:
        return m.depth + 1 <= acfg.max_depth and total_neurons(m.width, m.depth + 1) <= acfg.max_neurons

    # 3.1 Inner: optimize_width_at_fixed_depth
    def optimize_width_at_fixed_depth(curr_model: AE_UNET_STL) -> Tuple[AE_UNET_STL, float, Dict[str, Any]]:
        local_val, local_state = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
        local_best_val = local_val
        local_best_state = local_state
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        
        width_failure_count = 0
        
        while width_failure_count < acfg.trials_width:
            if not can_widen(curr_model):
                break
            
            next_model = expand_width(curr_model, acfg.ex_k, acfg.max_width, device)
            if next_model is None: 
                break
            curr_model = next_model
            
            v, s = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_state = s
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                width_failure_count = 0
                improvements.append((total_neurons(curr_model.width, curr_model.depth), v))
            else:
                width_failure_count += 1
                logger.log_console(f'[WIDTH OPT] ✗ No improvement')
                
        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        return final_model, local_best_val, local_best_snap

    # 4.1 Inner: optimize_depth_at_fixed_width
    def optimize_depth_at_fixed_width(curr_model: AE_UNET_STL) -> Tuple[AE_UNET_STL, float, Dict[str, Any]]:
        local_val, local_state = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
        local_best_val = local_val
        local_best_state = local_state
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        
        depth_failure_count = 0
        
        while depth_failure_count < acfg.trials_depth:
            if not can_deepen(curr_model):
                break
                
            next_model = expand_depth(curr_model, acfg.max_depth, device)
            if next_model is None:
                break
            curr_model = next_model
            
            v, s = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_state = s
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                depth_failure_count = 0
                improvements.append((total_neurons(curr_model.width, curr_model.depth), v))
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
            next_model = expand_depth(model, acfg.max_depth, device)
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
            next_model = expand_width(model, acfg.ex_k, acfg.max_width, device)
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
    # - Implemented forward-only logic for UNet.
    
    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"AE_UNet_STL ({acfg.adp_mode})")
    if log_neurons and improvements:
        plot_loss_vs_neurons([n for n,_ in improvements], [v for _,v in improvements], results_dir / "loss_vs_neurons.png", title=f"AE_UNet_STL ({acfg.adp_mode})")
        
    return global_best_val, model, model.width, model.depth


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
    p = argparse.ArgumentParser(description="ADP AE_UNet_STL width/depth search")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--pool-after", type=int, nargs="*", default=[])
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"])
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
    p.add_argument("--results-dir", type=Path, default=Path("results_adp_ae_unet"))
    p.add_argument("--plot-loss", action="store_true", help="Save loss-vs-epoch (log scale)")
    p.add_argument("--plot-neurons", action="store_true", help="Save neurons-vs-loss (log scale)")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE_UNET_STL(in_channels=3, width=args.width, depth=args.depth, pool_after=args.pool_after).to(device)
    acfg = ADPConfig(adp_mode=args.adp_mode, delta=args.delta, patience=args.patience, trials_width=args.trials_width,
                     trials_depth=args.trials_depth, ex_k=args.ex_k, max_width=args.max_width, max_depth=args.max_depth,
                     max_neurons=args.max_neurons, max_epochs=args.max_epochs, pool_after=args.pool_after)
    
    print(f"[ADP AE_UNet] Starting {args.adp_mode}, Init: Width={args.width}, Depth={args.depth}")
    best, model, w, d = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    print(f"[ADP AE_UNet] DONE. Best Val={best:.6f} Width={w} Depth={d}")
