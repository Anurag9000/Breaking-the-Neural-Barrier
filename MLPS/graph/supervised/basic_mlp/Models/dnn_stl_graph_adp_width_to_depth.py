import copy
from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore
from utils.adp_logging import ContinuousLogger
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Add root to sys.path for utils
sys.path.append(str(Path(__file__).resolve().parents[4]))
try:
    from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons
except ImportError:
    # Fallback if utils not found or different structure
    def plot_loss_vs_epoch(*args, **kwargs): pass
    def plot_loss_vs_neurons(*args, **kwargs): pass

from utils.adp_introspect import infer_adp_depth, infer_adp_shape, infer_adp_width, can_expand_depth, can_expand_width

# Load baseline
BASE_PATH = Path(__file__).with_name("dnn_stl_graph.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
ModelClass = baseline_module.DNNNodeFC

# ADP REVIEW (BEFORE REFACTOR)
# - This file is newly created to implement the ADP algorithms from scratch for the DNNNodeFC model.
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
    merged = {}
    for k, v in new_state.items():
        if k in old_state:
            ov = old_state[k]
            if ov.shape == v.shape:
                merged[k] = ov
            else:
                # Basic resizing - for complex models (MHA) this might need more spec
                # But for batch refactor we assume basic structure or compatible resizing
                if v.ndim == ov.ndim:
                    merged[k] = _resize_tensor(v.shape, ov)
                else:
                    merged[k] = v # mismatch dim, reset
        else:
            merged[k] = v
    return merged

def rebuild_model(model: ModelClass, width: int, depth: int, device, cfg: ADPConfig) -> ModelClass:
    def get_attr(obj, candidates, default):
        for c in candidates:
            try:
                val = obj
                for part in c.split('.'): val = getattr(val, part)
                return val
            except: continue
        return default
    try:
        kwargs = {}
        kwargs['num_nodes'] = get_attr(model, ['num_nodes'], 1)
        kwargs['num_classes'] = get_attr(model, ['num_classes', 'head.out_features', 'fc.out_features', 'classifier.out_features'], 1)
        kwargs['hidden'] = width
        kwargs['depth'] = depth
        new_model = ModelClass(**kwargs).to(device)
    except Exception as e:
        print(f'Rebuild failed: {e}')
        return None
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model
        
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model

def expand_width(model: ModelClass, ex_k: int, max_width: int, device, cfg: ADPConfig) -> Optional[ModelClass]:
    cur_w = width = getattr(model, 'hidden', 0) if 'hidden' != 'None' else getattr(model.cfg, 'width', 0) if hasattr(model, 'cfg') else 0
    new_w = min(cur_w + ex_k, max_width)
    if new_w == cur_w: return None
    return rebuild_model(model, new_w, getattr(model, 'depth', 1) if 'depth' != 'None' else 1, device, cfg)

def expand_depth(model: ModelClass, max_depth: int, device, cfg: ADPConfig) -> Optional[ModelClass]:
    cur = model.depth
    if cur >= max_depth:
        return None
    return rebuild_model(model, int(model.hidden), cur + 1, device, cfg)
    

def total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))

def snapshot_arch_and_state(model: ModelClass, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "width": int(getattr(model, "hidden", 0)),
        "depth": int(getattr(model, "depth", 0)),
        "state": copy.deepcopy(state)
    }

def restore_arch_and_state(model: ModelClass, snap: Dict[str, Any], device) -> ModelClass:
    restored = rebuild_model(model, int(snap["width"]), int(snap["depth"]), device, None)
    restored.load_state_dict(snap["state"], strict=False)
    return restored

def train_with_early_stopping(model: ModelClass, dl_train, dl_val, acfg: ADPConfig, device, history: list) -> Tuple[float, Dict[str, Any]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0
    
    # Basic training loop
    for epoch in range(1, acfg.max_epochs + 1):
        model.train()
        for batch in dl_train:
            if not isinstance(batch, (list, tuple)) or len(batch) != 5:
                raise ValueError("Expected graph supervision data tuple: (X, y, train_mask, val_mask, test_mask)")
            x, y, train_mask, _, _ = batch
            x = x.to(device)
            y = y.to(device)
            train_mask = train_mask.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            loss.backward()
            if acfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
        
        model.eval()
        val = 0.0
        with torch.no_grad():
            for batch in dl_val:
                if not isinstance(batch, (list, tuple)) or len(batch) != 5:
                    raise ValueError("Expected graph supervision data tuple: (X, y, train_mask, val_mask, test_mask)")
                x, y, _, val_mask, _ = batch
                x = x.to(device)
                y = y.to(device)
                val_mask = val_mask.to(device)
                logits = model(x)
                val += float(F.cross_entropy(logits[val_mask], y[val_mask]).item())
        val /= max(len(dl_val), 1)
        
        history.append(val)
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
            improved = True
        else:
            es_counter += 1
            improved = False

        # Log
        msg = f"  Epoch {epoch}/{acfg.max_epochs} | Val Loss: {val:.6f} | Best: {best_val:.6f} | ES: {es_counter}/{acfg.patience}"

        if logger:
             logger.log_console(msg)
             logger.log_epoch_stats({
                "epoch": epoch,
                "width": int(getattr(model, "hidden", 0)),
                "depth": int(getattr(model, "depth", 0)),
                "neurons": total_neurons(int(getattr(model, "hidden", 0)), int(getattr(model, "depth", 0))),
                "val_loss": val,
                "best_val": best_val,
                "es_counter": es_counter,
                "improved": improved
             })
        if es_counter >= acfg.patience: break
            
    return best_val, best_state

def adp_search(model: ModelClass, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp")):
    from utils.adp_contract import run_module_adp

    best_val, best_model = run_module_adp(
        globals(),
        model,
        dl_train,
        dl_val,
        acfg,
        device,
        log_loss=log_loss,
        log_neurons=log_neurons,
        results_dir=results_dir,
    )
    return best_val, best_model, *infer_adp_shape(best_model)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Cora", choices=["Cora","Citeseer","PubMed"])
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--adp-mode", default="width_to_depth", choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"])
    p.add_argument("--max-epochs", type=int, default=100000000)
    args = p.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, num_classes = load_planetoid(args.dataset)
    X, y, train_mask, val_mask, test_mask = data
    dl_train = [data]
    dl_val = [data]
    
    try:
        model = ModelClass(num_nodes=X.size(0), num_classes=num_classes, hidden=args.width, depth=args.depth).to(device)
    except:
        print("Could not instantiate model with default args.")
        return

    acfg = ADPConfig(adp_mode=args.adp_mode, max_epochs=args.max_epochs)
    val, m, w, d = adp_search(model, dl_train, dl_val, acfg, device)
    print(f"Done. Best val={val} w={w} d={d}")
