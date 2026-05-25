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
BASE_PATH = Path(__file__).with_name("ae_tied_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
AE_TIED_STL = baseline_module.AE_TIED_STL  # type: ignore
EncConvBlock = baseline_module.EncConvBlock  # type: ignore
DecTiedBlock = baseline_module.DecTiedBlock  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - Inner training: train_with_patience ties ES reset to delta and reloads immediately.
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - Control flow: toggles modes on no improvement; lacks forward-only march and context-end restore per updated spec.
# - ES patience conflated with expansion patiences; no snapshot/restore separation.

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


def rebuild_model(model: AE_TIED_STL, width: int, depth: int, device, pool_after: List[int]) -> AE_TIED_STL:
    new_model = AE_TIED_STL(in_channels=model.in_channels, width=width, depth=depth, pool_after=pool_after).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(model: AE_TIED_STL, ex_k: int, max_width: int, device) -> Optional[AE_TIED_STL]:
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return None
    return rebuild_model(model, new_w, model.depth, device, list(model.pool_after))


def expand_depth(model: AE_TIED_STL, max_depth: int, device) -> Optional[AE_TIED_STL]:
    if model.depth >= max_depth:
        return None
    return rebuild_model(model, model.width, model.depth + 1, device, list(model.pool_after))


def total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))


def snapshot_arch_and_state(model: AE_TIED_STL, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "width": model.width,
        "depth": model.depth,
        "pool_after": list(model.pool_after),
        "state": copy.deepcopy(state)
    }


def restore_arch_and_state(model: AE_TIED_STL, snap: Dict[str, Any], device) -> AE_TIED_STL:
    restored = AE_TIED_STL(
        in_channels=model.in_channels,
        width=snap["width"],
        depth=snap["depth"],
        pool_after=list(snap["pool_after"])
    ).to(device)
    restored.load_state_dict(snap["state"])
    return restored


def train_with_early_stopping(model: AE_TIED_STL, dl_train, dl_val, acfg: ADPConfig, device, history: list, logger: Optional[ContinuousLogger] = None, verbose: bool = True) -> Tuple[float, Dict[str, Any]]:
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
        
        if val < best_val: # Strict improvement for inner loop ES
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
                "epoch": len(history), # total cumulative epochs if history is global
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
            
    return best_val, best_state


def adp_search(model: AE_TIED_STL, dl_train, dl_val, acfg: ADPConfig, device, logger: ContinuousLogger, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp_tied_stl")):
    from utils.adp_contract import run_module_adp
    from utils.adp_introspect import infer_adp_shape

    best_val, model = run_module_adp(
        globals(),
        model,
        dl_train,
        dl_val,
        acfg,
        device,
        log_loss=locals().get("log_loss", False),
        log_neurons=locals().get("log_neurons", False),
        results_dir=locals().get("results_dir"),
        logger=locals().get("logger"),
    )

    return best_val, model, *infer_adp_shape(model)


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
    p = argparse.ArgumentParser(description="ADP Tied-weights AE width/depth search")
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
    p.add_argument("--results-dir", type=Path, default=Path("results_adp_tied_stl"))
    p.add_argument("--plot-loss", action="store_true", help="Save loss-vs-epoch (log scale)")
    p.add_argument("--plot-neurons", action="store_true", help="Save neurons-vs-loss (log scale)")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE_TIED_STL(in_channels=3, width=args.width, depth=args.depth, pool_after=args.pool_after).to(device)
    acfg = ADPConfig(adp_mode=args.adp_mode, delta=args.delta, patience=args.patience, trials_width=args.trials_width,
                     trials_depth=args.trials_depth, ex_k=args.ex_k, max_width=args.max_width, max_depth=args.max_depth,
                     max_neurons=args.max_neurons, max_epochs=args.max_epochs, pool_after=args.pool_after)
    
    # Initialize Logger
    logger = ContinuousLogger(args.results_dir, "ae_tied_stl", args.adp_mode)
    
    best, model, w, d = adp_search(model, dl_train, dl_val, acfg, device, logger=logger, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    logger.log_console(f"[ADP Tied AE] mode={args.adp_mode} best_val={best:.6f} width={w} depth={d}")
    logger.close()
