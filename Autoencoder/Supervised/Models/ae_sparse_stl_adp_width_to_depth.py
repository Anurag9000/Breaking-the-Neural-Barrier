import copy
from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore
from utils.adp_logging import ContinuousLogger

# Load baseline
BASE_PATH = Path(__file__).with_name("ae_sparse_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
AE_SPARSE_STL = baseline_module.AE_SPARSE_STL  # type: ignore
ConvBlock = baseline_module.ConvBlock  # type: ignore
DeconvBlock = baseline_module.DeconvBlock  # type: ignore
sparsity_penalty = baseline_module.sparsity_penalty  # type: ignore
ae_sparse_total_neurons = baseline_module.ae_sparse_total_neurons  # type: ignore

# ADP REVIEW (BEFORE REFACTOR)
# - Modes: width_only/width, depth_only/depth, width_to_depth, depth_to_width, alt_width, alt_depth share single loop with per-expansion rollback.
# - Inner training: train_with_patience ties ES reset to delta and reloads immediately.
# - Expansions: widen/deepen rollback on failure; shared delta/patience; no snapshot helpers.
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
    sparsity_mode: str = "l1"
    sparsity_rho: float = 0.05
    sparsity_lambda: float = 1e-3
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


def rebuild_model(model: AE_SPARSE_STL, width: int, depth: int, device, pool_after: List[int]) -> AE_SPARSE_STL:
    new_model = AE_SPARSE_STL(in_channels=model.in_channels, width=width, depth=depth, pool_after=pool_after).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def widen_model(model: AE_SPARSE_STL, ex_k: int, max_width: int, device):
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return None
    return rebuild_model(model, new_w, model.depth, device, list(model.pool_after))


def deepen_model(model: AE_SPARSE_STL, device):
    return rebuild_model(model, model.width, model.depth + 1, device, list(model.pool_after))


def snapshot_arch_and_state(model: AE_SPARSE_STL, state_dict=None):
    state = state_dict if state_dict is not None else model.state_dict()
    return {"width": model.width, "depth": model.depth, "pool_after": list(model.pool_after), "state": copy.deepcopy(state)}


def restore_arch_and_state(model: AE_SPARSE_STL, snap, device):
    restored = AE_SPARSE_STL(in_channels=model.in_channels, width=snap["width"], depth=snap["depth"], pool_after=list(snap["pool_after"])).to(device)
    restored.load_state_dict(snap["state"])
    return restored


def train_with_early_stopping(model: AE_SPARSE_STL, dl_train, dl_val, acfg: ADPConfig, device, history: list):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best = float("inf"); best_state=None; es_counter=0
    for _ in range(acfg.max_epochs):
        model.train()
        for x, _ in dl_train:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            rec, z = model(x)
            loss = F.mse_loss(rec, x) + acfg.sparsity_lambda * sparsity_penalty(z, mode=acfg.sparsity_mode, rho=acfg.sparsity_rho)
            loss.backward()
            if acfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = 0.0; n = 0
            for x, _ in dl_val:
                x = x.to(device)
                rec, z = model(x)
                l = F.mse_loss(rec, x) + acfg.sparsity_lambda * sparsity_penalty(z, mode=acfg.sparsity_mode, rho=acfg.sparsity_rho)
                val += l.item(); n += 1
            val = val / max(n,1)
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
        if val < best:
            best = val; best_state = copy.deepcopy(model.state_dict()); es_counter = 0
        else:
            es_counter += 1
        if es_counter >= acfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, best_state


def adp_search(model: AE_SPARSE_STL, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp_sparse_stl")):
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    improvements: List[tuple[int, float]] = []

    def can_widen(local_model: AE_SPARSE_STL):
        new_w = min(acfg.max_width, local_model.width + acfg.ex_k)
        return new_w > local_model.width and ae_sparse_total_neurons(new_w, local_model.depth) <= acfg.max_neurons

    def can_deepen(local_model: AE_SPARSE_STL):
        return local_model.depth + 1 <= acfg.max_depth and ae_sparse_total_neurons(local_model.width, local_model.depth + 1) <= acfg.max_neurons

    best_val, best_state = train_with_early_stopping(model, dl_train, dl_val, acfg, device, val_history, logger=logger)
    best_snap = snapshot_arch_and_state(model, best_state)
    best_width = model.width
    best_depth = model.depth
    improvements.append((ae_sparse_total_neurons(model.width, model.depth), best_val))
    pw, pd = acfg.trials_width, acfg.trials_depth

    def width_search(local_model: AE_SPARSE_STL, initial_val=None, initial_snap=None, log_improvement: bool = False):
        local_best_val = initial_val
        local_best_snap = initial_snap
        if local_best_val is None or local_best_snap is None:
            local_best_val, local_best_state = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            local_best_snap = snapshot_arch_and_state(local_model, local_best_state)
        width_failure_count = 0
        while width_failure_count < pw and can_widen(local_model):
            widened = widen_model(local_model, acfg.ex_k, acfg.max_width, device)
            if widened is None:
                break
            local_model = widened
            cand_val, cand_state = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            if cand_val < local_best_val - acfg.delta:
                local_best_val = cand_val
                local_best_snap = snapshot_arch_and_state(local_model, cand_state)
                width_failure_count = 0
                if log_improvement:
                    improvements.append((ae_sparse_total_neurons(local_model.width, local_model.depth), local_best_val))
            else:
                width_failure_count += 1
                logger.log_console(f'[WIDTH OPT] ✗ No improvement')
        local_model = restore_arch_and_state(local_model, local_best_snap, device)
        return local_model, local_best_val, local_best_snap

    def depth_search(local_model: AE_SPARSE_STL, initial_val=None, initial_snap=None, log_improvement: bool = False):
        local_best_val = initial_val
        local_best_snap = initial_snap
        if local_best_val is None or local_best_snap is None:
            local_best_val, local_best_state = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            local_best_snap = snapshot_arch_and_state(local_model, local_best_state)
        depth_failure_count = 0
        while depth_failure_count < pd and can_deepen(local_model):
            local_model = deepen_model(local_model, device)
            cand_val, cand_state = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history, logger=logger)
            if cand_val < local_best_val - acfg.delta:
                local_best_val = cand_val
                local_best_snap = snapshot_arch_and_state(local_model, cand_state)
                depth_failure_count = 0
                if log_improvement:
                    improvements.append((ae_sparse_total_neurons(local_model.width, local_model.depth), local_best_val))
            else:
                depth_failure_count += 1
                logger.log_console(f'[DEPTH OPT] ✗ No improvement')
        local_model = restore_arch_and_state(local_model, local_best_snap, device)
        return local_model, local_best_val, local_best_snap

    mode = acfg.adp_mode
    if mode in ("width_only", "width"):
        model, best_val, best_snap = width_search(model, initial_val=best_val, initial_snap=best_snap, log_improvement=True)
        best_width, best_depth = best_snap["width"], best_snap["depth"]
    elif mode in ("depth_only", "depth"):
        model, best_val, best_snap = depth_search(model, initial_val=best_val, initial_snap=best_snap, log_improvement=True)
        best_width, best_depth = best_snap["width"], best_snap["depth"]
    elif mode == "depth_to_width":
        model, best_val, best_snap = width_search(model, initial_val=best_val, initial_snap=best_snap, log_improvement=True)
        best_width, best_depth = best_snap["width"], best_snap["depth"]
        depth_failure_count = 0
        while depth_failure_count < pd and can_deepen(model):
            model = deepen_model(model, device)
            cand_model, cand_val, cand_snap = width_search(model)
            if cand_val < best_val - acfg.delta:
                best_val = cand_val
                best_snap = cand_snap
                best_width, best_depth = cand_snap["width"], cand_snap["depth"]
                depth_failure_count = 0
                model = restore_arch_and_state(model, cand_snap, device)
                improvements.append((ae_sparse_total_neurons(model.width, model.depth), best_val))
            else:
                model = cand_model
                depth_failure_count += 1
        model = restore_arch_and_state(model, best_snap, device)
    elif mode == "width_to_depth":
        model, best_val, best_snap = depth_search(model, initial_val=best_val, initial_snap=best_snap, log_improvement=True)
        best_width, best_depth = best_snap["width"], best_snap["depth"]
        width_failure_count = 0
        while width_failure_count < pw and can_widen(model):
            widened = widen_model(model, acfg.ex_k, acfg.max_width, device)
            if widened is None:
                break
            model = widened
            cand_model, cand_val, cand_snap = depth_search(model)
            if cand_val < best_val - acfg.delta:
                best_val = cand_val
                best_snap = cand_snap
                best_width, best_depth = cand_snap["width"], cand_snap["depth"]
                width_failure_count = 0
                model = restore_arch_and_state(model, cand_snap, device)
                improvements.append((ae_sparse_total_neurons(model.width, model.depth), best_val))
            else:
                model = cand_model
                width_failure_count += 1
        model = restore_arch_and_state(model, best_snap, device)
    elif mode == "alt_depth":
        depth_saturated = False
        width_saturated = False
        phase = "depth"
        while not (depth_saturated and width_saturated):
            if phase == "depth":
                model = restore_arch_and_state(model, best_snap, device)
                cand_model, cand_val, cand_snap = depth_search(model, initial_val=best_val, initial_snap=best_snap)
                if cand_val < best_val - acfg.delta:
                    best_val = cand_val
                    best_snap = cand_snap
                    best_width, best_depth = cand_snap["width"], cand_snap["depth"]
                    depth_saturated = False
                    improvements.append((ae_sparse_total_neurons(cand_model.width, cand_model.depth), best_val))
                else:
                    depth_saturated = True
                model = restore_arch_and_state(model, best_snap, device)
                phase = "width"
            else:
                model = restore_arch_and_state(model, best_snap, device)
                cand_model, cand_val, cand_snap = width_search(model, initial_val=best_val, initial_snap=best_snap)
                if cand_val < best_val - acfg.delta:
                    best_val = cand_val
                    best_snap = cand_snap
                    best_width, best_depth = cand_snap["width"], cand_snap["depth"]
                    width_saturated = False
                    improvements.append((ae_sparse_total_neurons(cand_model.width, cand_model.depth), best_val))
                else:
                    width_saturated = True
                model = restore_arch_and_state(model, best_snap, device)
                phase = "depth"
    elif mode == "alt_width":
        depth_saturated = False
        width_saturated = False
        phase = "width"
        while not (depth_saturated and width_saturated):
            if phase == "width":
                model = restore_arch_and_state(model, best_snap, device)
                cand_model, cand_val, cand_snap = width_search(model, initial_val=best_val, initial_snap=best_snap)
                if cand_val < best_val - acfg.delta:
                    best_val = cand_val
                    best_snap = cand_snap
                    best_width, best_depth = cand_snap["width"], cand_snap["depth"]
                    width_saturated = False
                    improvements.append((ae_sparse_total_neurons(cand_model.width, cand_model.depth), best_val))
                else:
                    width_saturated = True
                model = restore_arch_and_state(model, best_snap, device)
                phase = "depth"
            else:
                model = restore_arch_and_state(model, best_snap, device)
                cand_model, cand_val, cand_snap = depth_search(model, initial_val=best_val, initial_snap=best_snap)
                if cand_val < best_val - acfg.delta:
                    best_val = cand_val
                    best_snap = cand_snap
                    best_width, best_depth = cand_snap["width"], cand_snap["depth"]
                    depth_saturated = False
                    improvements.append((ae_sparse_total_neurons(cand_model.width, cand_model.depth), best_val))
                else:
                    depth_saturated = True
                model = restore_arch_and_state(model, best_snap, device)
                phase = "width"
    else:
        raise ValueError(f"Unsupported ADP mode: {mode}")

    model = restore_arch_and_state(model, best_snap, device)
    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    if log_neurons and improvements:
        plot_loss_vs_neurons([n for n,_ in improvements], [v for _,v in improvements], results_dir / "loss_vs_neurons.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    return best_val, model, best_snap["width"], best_snap["depth"]


# ADP REVIEW (AFTER REFACTOR)
# - width_only/width -> ADP_WIDTH_ONLY: forward-only widening with width_failure_count < trials_width; restore best snapshot at end.
# - depth_only/depth -> ADP_DEPTH_ONLY: forward-only deepening with depth_failure_count < trials_depth; restore best snapshot at end.
# - depth_to_width -> ADP_DEPTH_OUTER_WIDTH_INNER: outer depth marches forward; inner width_search forward-only; accept on delta improvement; restore global best after outer loop.
# - width_to_depth -> ADP_WIDTH_OUTER_DEPTH_INNER: outer width marches forward; inner depth_search forward-only; accept on delta improvement; restore global best after outer loop.
# - alt_depth/alt_width -> Alternating phases starting with depth or width; each phase forward-only on that dimension, starting from global best and restoring it at phase end; stop when both dimensions saturate.


def make_loaders(batch_size: int = 128, val_split: float = 0.1):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_train, dl_val


def main():
    import argparse
    p = argparse.ArgumentParser(description="ADP Sparse AE width/depth search")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--pool-after", type=int, nargs="*", default=[])
    p.add_argument("--sparsity-mode", type=str, default="l1", choices=["l1","kl"])
    p.add_argument("--sparsity-rho", type=float, default=0.05)
    p.add_argument("--sparsity-lambda", type=float, default=1e-3)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only","depth_only","width_to_depth","depth_to_width","alt_width","alt_depth","width","depth"])
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
    p.add_argument("--results-dir", type=Path, default=Path("results_adp_sparse_stl"))
    p.add_argument("--plot-loss", action="store_true", help="Save loss-vs-epoch (log scale)")
    p.add_argument("--plot-neurons", action="store_true", help="Save neurons-vs-loss (log scale)")
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE_SPARSE_STL(in_channels=3, width=args.width, depth=args.depth, pool_after=args.pool_after).to(device)
    acfg = ADPConfig(adp_mode=args.adp_mode, delta=args.delta, patience=args.patience, trials_width=args.trials_width,
                     trials_depth=args.trials_depth, ex_k=args.ex_k, max_width=args.max_width, max_depth=args.max_depth,
                     max_neurons=args.max_neurons, max_epochs=args.max_epochs, sparsity_mode=args.sparsity_mode,
                     sparsity_rho=args.sparsity_rho, sparsity_lambda=args.sparsity_lambda, pool_after=args.pool_after)
    best_val, model, w, d = adp_search(model, dl_train, dl_val, acfg, device, log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    print(f"[ADP Sparse AE] mode={args.adp_mode} best_val={best_val:.6f} width={w} depth={d}")


if __name__ == "__main__":
    main()
