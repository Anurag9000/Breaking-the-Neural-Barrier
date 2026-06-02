import copy
from dataclasses import dataclass
import importlib.util

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add root to sys.path for utils
sys.path.append(str(Path(__file__).resolve().parents[4]))
try:
    from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons
    from utils.adp_state import merge_state_preserving_init
except ImportError:
    # Fallback if utils not found or different structure
    def plot_loss_vs_epoch(*args, **kwargs): pass
    def plot_loss_vs_neurons(*args, **kwargs): pass
    def merge_state_preserving_init(new_state, old_state):
        return new_state
from utils.adp_logging import ContinuousLogger

# Load baseline
BASE_PATH = Path(__file__).with_name("CNN_STL.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
ModelClass = baseline_module.ConvNetSTL

# ADP REVIEW (BEFORE REFACTOR)
# - This file is newly created to implement the ADP algorithms from scratch for the ConvBNReLU model.
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
    max_depth: int = 5
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    max_epochs: int = 100_000_000
    # Dynamic args
    

def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt

def _merge_state(new_state, old_state):
    return merge_state_preserving_init(new_state, old_state)

def _model_width(model, default=0):
    for obj in (model, getattr(model, "cfg", None)):
        if obj is None:
            continue
        for key in ("width", "channels", "out_ch", "out_channels", "planes"):
            if hasattr(obj, key):
                val = getattr(obj, key)
                if val is not None:
                    return val
    return default


def _model_depth(model, default=1):
    for obj in (model, getattr(model, "cfg", None)):
        if obj is None:
            continue
        for key in ("depth", "layers"):
            if hasattr(obj, key):
                val = getattr(obj, key)
                if val is not None:
                    return val
    return default


def _model_neurons(model):
    return total_neurons(_model_width(model), _model_depth(model))

def rebuild_model(model: ModelClass, width: int, depth: int, device, cfg: ADPConfig) -> ModelClass:
    try:
        new_model = ModelClass(
            input_channels=model.input_channels,
            num_classes=model.num_classes,
            width=width,
            depth=depth,
            pooling_indices=model.pooling_indices
        ).to(device)
    except Exception as e:
        print(f"Rebuild failed: {e}")
        return None
        
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    new_model.width = width
    new_model.depth = depth
    return new_model

def expand_width(model: ModelClass, ex_k: int, max_width: int, device, cfg: ADPConfig) -> Optional[ModelClass]:
    new_w = min(model.width + ex_k, max_width)
    if new_w == model.width and ex_k > 0: return None
    if new_w > max_width: return None
    return rebuild_model(model, new_w, model.depth, device, cfg)

def expand_depth(model: ModelClass, max_depth: int, device, cfg: ADPConfig) -> Optional[ModelClass]:
    new_d = min(model.depth + 1, max_depth)
    if new_d == model.depth: return None
    return rebuild_model(model, model.width, new_d, device, cfg)

def total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))

def snapshot_arch_and_state(model: ModelClass, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "width": model.width,
        "depth": model.depth,
        "pooling_indices": model.pooling_indices,
        "input_channels": model.input_channels,
        "num_classes": model.num_classes,
        "state": copy.deepcopy(state)
    }

def restore_arch_and_state(model: ModelClass, snap: Dict[str, Any], device) -> ModelClass:
    try:
        new_model = ModelClass(
            input_channels=snap.get("input_channels", 3),
            num_classes=snap.get("num_classes", 10),
            width=snap["width"],
            depth=snap["depth"],
            pooling_indices=snap.get("pooling_indices", ())
        ).to(device)
        new_model.load_state_dict(snap["state"])
        new_model.width = snap.get("width", getattr(new_model, "width", 0))
        new_model.depth = snap.get("depth", getattr(new_model, "depth", 1))
        return new_model
    except Exception:
        return model

def train_with_early_stopping(model: ModelClass, dl_train, dl_val, acfg: ADPConfig, device, history: list, logger: Optional[ContinuousLogger] = None, verbose: bool = True) -> Tuple[float, Dict[str, Any]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=acfg.max_epochs, eta_min=acfg.lr * 1e-2)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0
    
    # Basic training loop
    for _ in range(acfg.max_epochs):
        model.train()
        # Handle different dataset outputs (tuple vs single)
        for batch in dl_train:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
                # simple autoencoder or supervised assumption
                if len(batch) > 1:
                    y = batch[1].to(device)
                else:
                    y = x
            else:
                x = batch.to(device)
                y = x
                
            opt.zero_grad(set_to_none=True)
            try:
                out = model(x)
                # Handle tuple output
                if isinstance(out, (list, tuple)):
                    rec = out[0]
                else:
                    rec = out
                
                loss = F.cross_entropy(rec, y)
            except Exception:
                raise
            
            loss.backward()
            opt.step()
        
        # Validation
        model.eval()
        val = 0.0
        n = 0
        with torch.no_grad():
             for batch in dl_val:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                    y = batch[1].to(device) if len(batch)>1 else x
                else:
                    x = batch.to(device)
                    y = x
                
                try: 
                    out = model(x)
                    if isinstance(out, (list, tuple)): rec = out[0]
                    else: rec = out
                    
                    l = F.cross_entropy(rec, y)
                    
                    val += l.item()
                    n += 1
                except: pass
        if n>0: val /= n
        
        history.append(val)
        if val < best_val - acfg.delta:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
            improved_str = " ✓ NEW BEST"
        else:
            es_counter += 1
            improved_str = ""

        # Log to console and text file
        msg = f"  Epoch {_+1}/{acfg.max_epochs} | Device: {device} | Val Loss: {val:.6f} | Best: {best_val:.6f} | ES: {es_counter}/{acfg.patience}{improved_str}"
        if verbose and logger:
            logger.log_console(msg)
        elif verbose:
            print(msg)
        
        # Log to CSV immediately
        if logger:
            logger.log_epoch_stats({
                "epoch": len(history),
                "width": model.width,
                "depth": model.depth,
                "neurons": total_neurons(model.width, model.depth),
                "val_loss": val,
                "best_val": best_val,
                "es_counter": es_counter,
                "improved": bool(improved_str)
            })
        if es_counter >= acfg.patience:
            break

        # Cosine LR step per epoch
        scheduler.step()
            
    return best_val, best_state

def adp_search(model: ModelClass, dl_train, dl_val, acfg: ADPConfig, device, logger: ContinuousLogger, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp")):
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    improvements: List[tuple[int, float]] = []

    # Initial training
    logger.log_console(f"[INITIAL TRAINING]")
    best_val, best_state = train_with_early_stopping(model, dl_train, dl_val, acfg, device, val_history, logger=logger)
    model.load_state_dict(best_state)
    global_best_snap = snapshot_arch_and_state(model, best_state)
    global_best_val = best_val
    improvements.append((total_neurons(model.width, model.depth), best_val))

    def can_widen(m: ModelClass) -> bool:
        return (m.width + acfg.ex_k <= acfg.max_width) and (total_neurons(m.width + acfg.ex_k, m.depth) <= acfg.max_neurons)

    def can_deepen(m: ModelClass) -> bool:
        return (m.depth + 1 <= acfg.max_depth) and (total_neurons(m.width, m.depth + 1) <= acfg.max_neurons)

    def optimize_width_at_fixed_depth(curr_model: ModelClass) -> Tuple[ModelClass, float, Dict[str, Any]]:
        local_val, local_state = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
        local_best_val = local_val
        local_best_state = local_state
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        width_failure_count = 0
        while width_failure_count < acfg.trials_width:
            if not can_widen(curr_model): break
            next_model = expand_width(curr_model, acfg.ex_k, acfg.max_width, device, acfg)
            if next_model is None: break
            curr_model = next_model
            v, s = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_state = s
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                width_failure_count = 0
                improvements.append((total_neurons(curr_model.width, curr_model.depth), v))
                logger.log_console(f"[WIDTH OPT] ✓ IMPROVEMENT: New best: {v:.6f}")
                if log_loss: plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
                if log_neurons: plot_loss_vs_neurons([n for n,_ in improvements], [v for _,v in improvements], results_dir / "loss_vs_neurons.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
            else:
                width_failure_count += 1
                logger.log_console(f"[WIDTH OPT] ✗ No improvement")
        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        return final_model, local_best_val, local_best_snap

    def optimize_depth_at_fixed_width(curr_model: ModelClass) -> Tuple[ModelClass, float, Dict[str, Any]]:
        local_val, local_state = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
        local_best_val = local_val
        local_best_state = local_state
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        depth_failure_count = 0
        while depth_failure_count < acfg.trials_depth:
            if not can_deepen(curr_model): break
            next_model = expand_depth(curr_model, acfg.max_depth, device, acfg)
            if next_model is None: break
            curr_model = next_model
            v, s = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_state = s
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                depth_failure_count = 0
                improvements.append((total_neurons(curr_model.width, curr_model.depth), v))
                logger.log_console(f"[DEPTH OPT] ✓ IMPROVEMENT: New best: {v:.6f}")
                if log_loss: plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
                if log_neurons: plot_loss_vs_neurons([n for n,_ in improvements], [v for _,v in improvements], results_dir / "loss_vs_neurons.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
            else:
                depth_failure_count += 1
                logger.log_console(f"[DEPTH OPT] ✗ No improvement")
        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        return final_model, local_best_val, local_best_snap

    mode = acfg.adp_mode
    if mode in ["width_only", "width"]:
        model, global_best_val, global_best_snap = optimize_width_at_fixed_depth(model)
    elif mode in ["depth_only", "depth"]:
        model, global_best_val, global_best_snap = optimize_depth_at_fixed_width(model)
    elif mode == "width_to_depth":
        model, base_val, base_snap = optimize_width_at_fixed_depth(model)
        global_best_val = base_val
        global_best_snap = base_snap
        fc = 0
        while fc < acfg.trials_depth:
            if not can_deepen(model): break
            nm = expand_depth(model, acfg.max_depth, device, acfg)
            if nm is None: break
            model = nm
            model, v, s = optimize_width_at_fixed_depth(model)
            if v < global_best_val - acfg.delta:
                global_best_val = v
                global_best_snap = s
                fc = 0
            else: fc += 1
        model = restore_arch_and_state(model, global_best_snap, device)
    elif mode == "depth_to_width":
        model, base_val, base_snap = optimize_depth_at_fixed_width(model)
        global_best_val = base_val
        global_best_snap = base_snap
        fc = 0
        while fc < acfg.trials_width:
            if not can_widen(model): break
            nm = expand_width(model, acfg.ex_k, acfg.max_width, device, acfg)
            if nm is None: break
            model = nm
            model, v, s = optimize_depth_at_fixed_width(model)
            if v < global_best_val - acfg.delta:
                global_best_val = v
                global_best_snap = s
                fc = 0
            else: fc += 1
        model = restore_arch_and_state(model, global_best_snap, device)
    elif mode in ["alt_width", "alt_depth"]:
        phase = "width" if mode == "alt_width" else "depth"
        sat_w, sat_d = False, False
        while not (sat_w and sat_d):
            imp = False
            if phase == "width":
                model, v, s = optimize_width_at_fixed_depth(model)
                if v < global_best_val - acfg.delta:
                    global_best_val = v
                    global_best_snap = s
                    imp = True
                sat_w = not imp
                phase = "depth"
            else:
                model, v, s = optimize_depth_at_fixed_width(model)
                if v < global_best_val - acfg.delta:
                    global_best_val = v
                    global_best_snap = s
                    imp = True
                sat_d = not imp
                phase = "width"
            model = restore_arch_and_state(model, global_best_snap, device)
        model = restore_arch_and_state(model, global_best_snap, device)
    
    if log_loss: plot_loss_vs_epoch(val_history, results_dir / "loss.png")

    return global_best_val, model, model.width, model.depth

def make_loaders(
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 0,
    use_augment: bool = True,
    data_root: str = "./data",
):
    """
    CIFAR-10 loaders with optional augmentation/normalization.
    Augment adds RandomCrop+Flip; all splits get channel-wise normalization.
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)
    aug = []
    if use_augment:
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train_tf = transforms.Compose([*aug, transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)])

    ds_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    ds_eval = datasets.CIFAR10(root=data_root, train=True, download=False, transform=eval_tf)

    n_val = int(len(ds_train) * val_split)
    n_train = len(ds_train) - n_val
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(ds_train), generator=generator).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_ds = Subset(ds_train, train_idx)
    val_ds = Subset(ds_eval, val_idx)

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_train, dl_val

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--adp-mode", default="width_to_depth", choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"])
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--results-dir", type=str, default="results_adp_cnn_stl")
    
    # ADP Args
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--no-augment", action="store_true", help="Disable CIFAR augmentation (crop/flip)")
    
    # Plotting flags
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")

    args = p.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Real data
    dl_train, dl_val = make_loaders(
        batch_size=args.batch_size,
        use_augment=not args.no_augment,
        data_root=args.data_root,
    )
    
    model = ModelClass(input_channels=3, num_classes=10, width=args.width, depth=args.depth).to(device)

    acfg = ADPConfig(
        adp_mode=args.adp_mode, 
        max_epochs=args.max_epochs,
        delta=args.delta,
        patience=args.patience,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
        ex_k=args.ex_k,
        max_width=args.max_width,
        max_depth=args.max_depth,
        max_neurons=args.max_neurons,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip
    )
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Logger
    logger = ContinuousLogger(results_dir, "cnn_stl", args.adp_mode)
    
    val, m, w, d = adp_search(model, dl_train, dl_val, acfg, device, logger=logger, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=results_dir)
    logger.log_console(f"Done. Best val={val} w={w} d={d}")
    logger.close()
