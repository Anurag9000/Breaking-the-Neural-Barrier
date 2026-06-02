import copy
from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

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

# Load baseline
BASE_PATH = Path(__file__).with_name("cnn_vgg.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
ModelClass = baseline_module.VGG

# ADP REVIEW (BEFORE REFACTOR)
# - This file is newly created to implement the ADP algorithms from scratch for the VGG model.
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

# ADP wrapper for VGG
# We map 'width' to base_channels (usually 64)
# We map 'depth' to total number of convolutional layers

def generate_vgg_cfg(width: int, depth: int) -> List[Any]:
    # 5 stages for VGG on 32x32 (approx)
    # Channels: w, 2w, 4w, 8w, 8w
    # We distribute 'depth' layers across 5 stages.
    # Ensure at least 1 layer per stage
    depth = max(depth, 5)
    
    # Simple distribution: remainder added to later stages or middle?
    # VGG-16 (13 convs): 2, 2, 3, 3, 3
    # VGG-11 (8 convs): 1, 1, 2, 2, 2
    
    base_counts = [depth // 5] * 5
    rem = depth % 5
    # Add remainder to last stages (features) or middle?
    # VGG convention adds remainder to later stages.
    for i in range(rem):
        base_counts[4 - i] += 1
        
    cfg = []
    # Stage 1
    for _ in range(base_counts[0]): cfg.append(width)
    cfg.append('M')
    # Stage 2
    for _ in range(base_counts[1]): cfg.append(width*2)
    cfg.append('M')
    # Stage 3
    for _ in range(base_counts[2]): cfg.append(width*4)
    cfg.append('M')
    # Stage 4
    for _ in range(base_counts[3]): cfg.append(width*8)
    cfg.append('M')
    # Stage 5
    for _ in range(base_counts[4]): cfg.append(width*8)
    cfg.append('M')
    
    return cfg

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
        new_vgg_cfg = generate_vgg_cfg(width, depth)
        
        # Preserve other attributes
        num_classes = getattr(model, 'num_classes', 10)
        in_channels = getattr(model, 'in_channels', 3)
        bn = getattr(model, 'batch_norm', True)
        dropout = getattr(model, 'dropout', 0.1)

        new_model = ModelClass(
            cfg=new_vgg_cfg,
            num_classes=num_classes,
            in_channels=in_channels,
            batch_norm=bn,
            dropout=dropout
        ).to(device)
        new_model.width = width
        new_model.depth = depth
    except Exception as e:
        print(f"Rebuild failed: {e}")
        return None

    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    new_model.width = width
    new_model.depth = depth
    return new_model

def expand_width(model: ModelClass, ex_k: int, max_width: int, device, cfg: ADPConfig) -> Optional[ModelClass]:
    # Estimate current width from first layer
    cur_cfg = model.cfg
    # Find first int
    cur_w = 64
    for x in cur_cfg:
        if isinstance(x, int):
            cur_w = x
            break
            
    # Typically VGG width steps are powers of 2 or +k?
    # ex_k is usually small(16).
    new_w = min(cur_w + ex_k, max_width)
    if new_w <= cur_w: return None
    
    # Count depth
    cur_d = sum(1 for x in model.cfg if isinstance(x, int))
    
    return rebuild_model(model, new_w, cur_d, device, cfg)

def expand_depth(model: ModelClass, max_depth: int, device, cfg: ADPConfig) -> Optional[ModelClass]:
    cur_d = sum(1 for x in model.cfg if isinstance(x, int))
    if cur_d >= max_depth: return None
    
    # Width
    cur_w = 64
    for x in model.cfg:
        if isinstance(x, int):
            cur_w = x
            break
            
    return rebuild_model(model, cur_w, cur_d + 1, device, cfg)

def total_neurons(width: int, depth: int) -> int:
    # Approx
    return width * depth # scalable proxy

def snapshot_arch_and_state(model: ModelClass, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    cur_w = 64
    if hasattr(model, 'cfg'):
        for x in model.cfg:
            if isinstance(x, int):
                cur_w = x
                break
    cur_d = sum(1 for x in model.cfg if isinstance(x, int)) if hasattr(model, 'cfg') else 0
    
    return {
        "width": cur_w,
        "depth": cur_d,
        "state": copy.deepcopy(state)
    }

def restore_arch_and_state(model: ModelClass, snap: Dict[str, Any], device) -> ModelClass:
    return rebuild_model(model, snap['width'], snap['depth'], device, None)

def train_with_early_stopping(model: ModelClass, dl_train, dl_val, acfg: ADPConfig, device, history: list) -> Tuple[float, Dict[str, Any]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0
    loss_fn = nn.CrossEntropyLoss()
    
    for _ in range(acfg.max_epochs):
        model.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss += loss.item()
                n += 1
        if n > 0: val_loss /= n
        
        history.append(val_loss)
        if val_loss < best_val - acfg.delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1
        if es_counter >= acfg.patience: break
        
    return best_val, best_state

# Helper for data
def make_loaders(batch_size=128):
    from torchvision import datasets, transforms
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    t_test = transforms.Compose([transforms.ToTensor()])
    
    try:
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=t_train)
        evalset = datasets.CIFAR10(root='./data', train=True, download=False, transform=t_test)
        # Split train deterministically and keep validation on the eval transform.
        n = len(trainset)
        n_val_split = n // 10
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(n, generator=g).tolist()
        train_idx = indices[: n - n_val_split]
        val_idx = indices[n - n_val_split :]
        trainset = Subset(trainset, train_idx)
        valset = Subset(evalset, val_idx)
    except Exception as exc:
        raise RuntimeError(
            "CIFAR10 benchmark data is unavailable. Download the real dataset before running this benchmark."
        ) from exc
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

def adp_search(model: ModelClass, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp")):
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    improvements: List[tuple[int, float]] = []

    best_val, best_state = train_with_early_stopping(model, dl_train, dl_val, acfg, device, val_history)
    model.load_state_dict(best_state)
    global_best_snap = snapshot_arch_and_state(model, best_state)
    global_best_val = best_val
    improvements.append((total_neurons(model.width, model.depth), best_val))

    def can_widen(m: ModelClass) -> bool:
        return (m.width + acfg.ex_k <= acfg.max_width) and (total_neurons(m.width + acfg.ex_k, m.depth) <= acfg.max_neurons)

    def can_deepen(m: ModelClass) -> bool:
        return (m.depth + 1 <= acfg.max_depth) and (total_neurons(m.width, m.depth + 1) <= acfg.max_neurons)

    def optimize_width_at_fixed_depth(curr_model: ModelClass):
        local_val, local_state = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history)
        local_best_val = local_val
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        failure_count = 0
        while failure_count < acfg.trials_width:
            if not can_widen(curr_model):
                break
            next_model = expand_width(curr_model, acfg.ex_k, acfg.max_width, device, acfg)
            if next_model is None:
                break
            curr_model = next_model
            v, s = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history)
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                failure_count = 0
                improvements.append((total_neurons(curr_model.width, curr_model.depth), v))
            else:
                failure_count += 1
        return restore_arch_and_state(curr_model, local_best_snap, device), local_best_val, local_best_snap

    def optimize_depth_at_fixed_width(curr_model: ModelClass):
        local_val, local_state = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history)
        local_best_val = local_val
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        failure_count = 0
        while failure_count < acfg.trials_depth:
            if not can_deepen(curr_model):
                break
            next_model = expand_depth(curr_model, acfg.max_depth, device, acfg)
            if next_model is None:
                break
            curr_model = next_model
            v, s = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history)
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                failure_count = 0
                improvements.append((total_neurons(curr_model.width, curr_model.depth), v))
            else:
                failure_count += 1
        return restore_arch_and_state(curr_model, local_best_snap, device), local_best_val, local_best_snap

    mode = acfg.adp_mode
    if mode in ["width_only", "width"]:
        model, global_best_val, global_best_snap = optimize_width_at_fixed_depth(model)
    elif mode in ["depth_only", "depth"]:
        model, global_best_val, global_best_snap = optimize_depth_at_fixed_width(model)
    elif mode == "width_to_depth":
        model, global_best_val, global_best_snap = optimize_width_at_fixed_depth(model)
        fail = 0
        while fail < acfg.trials_depth:
            if not can_deepen(model):
                break
            deeper = expand_depth(model, acfg.max_depth, device, acfg)
            if deeper is None:
                break
            model, v, global_best_snap = optimize_depth_at_fixed_width(deeper)
            if v < global_best_val - acfg.delta:
                global_best_val = v
                fail = 0
            else:
                fail += 1
    elif mode == "depth_to_width":
        model, global_best_val, global_best_snap = optimize_depth_at_fixed_width(model)
        fail = 0
        while fail < acfg.trials_width:
            if not can_widen(model):
                break
            wider = expand_width(model, acfg.ex_k, acfg.max_width, device, acfg)
            if wider is None:
                break
            model, v, global_best_snap = optimize_width_at_fixed_depth(wider)
            if v < global_best_val - acfg.delta:
                global_best_val = v
                fail = 0
            else:
                fail += 1
    elif mode in ["alt_width", "alt_depth"]:
        phase = "width" if mode == "alt_width" else "depth"
        sat_w, sat_d = False, False
        while not (sat_w and sat_d):
            improved = False
            if phase == "width":
                model, v, global_best_snap = optimize_width_at_fixed_depth(model)
                if v < global_best_val - acfg.delta:
                    global_best_val = v
                    improved = True
                sat_w = not improved
                phase = "depth"
            else:
                model, v, global_best_snap = optimize_depth_at_fixed_width(model)
                if v < global_best_val - acfg.delta:
                    global_best_val = v
                    improved = True
                sat_d = not improved
                phase = "width"
    final_model = restore_arch_and_state(model, global_best_snap, device)
    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"VGG ({mode})")
    if log_neurons and improvements:
        plot_loss_vs_neurons([n for n, _ in improvements], [v for _, v in improvements], results_dir / "loss_vs_neurons.png", title=f"VGG ({mode})")
    return global_best_val, final_model, final_model.width, final_model.depth

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=8) # VGG-11 approx
    p.add_argument("--adp-mode", default="width_to_depth", choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"])
    p.add_argument("--max-epochs", type=int, default=10)
    args = p.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading data...")
    dl_train, dl_val = make_loaders()
    
    # Initial model
    # Generate cfg from args
    cfg_list = generate_vgg_cfg(args.width, args.depth)
    model = ModelClass(cfg=cfg_list, num_classes=10, batch_norm=True).to(device)
    model.width = args.width
    model.depth = sum(1 for x in cfg_list if isinstance(x, int))
    
    acfg = ADPConfig(adp_mode=args.adp_mode, max_epochs=args.max_epochs)
    val, m, w, d = adp_search(model, dl_train, dl_val, acfg, device)
    print(f"Done. Best val={val} w={w} d={d}")
