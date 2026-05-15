import copy
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger  # type: ignore
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

from .dae_graph_nodefeat_sup_stl import SupDAEGraphNodeFeat, sup_dae_total_neurons
from ..Runs.run_dae_graph_nodefeat_sup_stl import (
    build_dataset,
    add_gaussian_noise,
    LabeledGraphDataset,
)


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 256
    max_depth: int = 8
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 200
    noise_std: float = 0.1
    lambda_recon: float = 1.0


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt


def _merge_state(new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for k, v in new_state.items():
        if k in old_state:
            ov = old_state[k]
            merged[k] = ov if ov.shape == v.shape else _resize_tensor(v.shape, ov)
        else:
            merged[k] = v
    return merged


def rebuild_model(
    model: SupDAEGraphNodeFeat,
    in_dim: int,
    num_classes: int,
    width: int,
    depth: int,
    device: torch.device,
) -> SupDAEGraphNodeFeat:
    new_model = SupDAEGraphNodeFeat(in_dim=in_dim, num_classes=num_classes, width=width, depth=depth).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(
    model: SupDAEGraphNodeFeat,
    in_dim: int,
    num_classes: int,
    ex_k: int,
    max_width: int,
    device: torch.device,
) -> Optional[SupDAEGraphNodeFeat]:
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return None
    return rebuild_model(model, in_dim, num_classes, new_w, model.depth, device)


def expand_depth(
    model: SupDAEGraphNodeFeat,
    in_dim: int,
    num_classes: int,
    max_depth: int,
    device: torch.device,
) -> Optional[SupDAEGraphNodeFeat]:
    if model.depth >= max_depth:
        return None
    return rebuild_model(model, in_dim, num_classes, model.width, model.depth + 1, device)


def snapshot_arch_and_state(model: SupDAEGraphNodeFeat, state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    st = state if state is not None else model.state_dict()
    return {
        "in_dim": model.in_dim,
        "width": model.width,
        "depth": model.depth,
        "num_classes": model.num_classes,
        "state": copy.deepcopy(st),
    }


def restore_arch_and_state(snap: Dict[str, Any], device: torch.device) -> SupDAEGraphNodeFeat:
    mdl = SupDAEGraphNodeFeat(
        in_dim=snap["in_dim"],
        num_classes=snap["num_classes"],
        width=snap["width"],
        depth=snap["depth"],
    ).to(device)
    mdl.load_state_dict(snap["state"], strict=False)
    return mdl


def train_val_one_pass(
    model: SupDAEGraphNodeFeat,
    dl: DataLoader,
    opt: torch.optim.Optimizer,
    acfg: ADPConfig,
    device: torch.device,
) -> Tuple[float, float]:
    """Train on train_mask and evaluate val loss on val_mask in a single graph."""
    model.train()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    total_train, total_nodes = 0.0, 0

    for x, adj, y, train_mask, val_mask, _ in dl:
        x = x.squeeze(0).to(device)
        adj = adj.squeeze(0).to(device)
        y = y.squeeze(0).to(device)
        train_mask = train_mask.squeeze(0).to(device)
        val_mask = val_mask.squeeze(0).to(device)

        # Train step
        x_noisy = add_gaussian_noise(x, acfg.noise_std)
        opt.zero_grad(set_to_none=True)
        x_rec, logits = model(x_noisy, adj)
        loss_recon = mse(x_rec, x)
        if train_mask.sum() > 0:
            loss_cls = ce(logits[train_mask], y[train_mask])
        else:
            loss_cls = torch.tensor(0.0, device=device)
        loss = acfg.lambda_recon * loss_recon + loss_cls
        loss.backward()
        if acfg.grad_clip and acfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
        opt.step()

        n = max(int(train_mask.sum().item()), 1)
        total_train += float(loss.item()) * n
        total_nodes += n

        # Validation loss on val_mask
        model.eval()
        with torch.no_grad():
            x_noisy_val = add_gaussian_noise(x, acfg.noise_std)
            x_rec_val, logits_val = model(x_noisy_val, adj)
            loss_recon_val = mse(x_rec_val, x) / x.size(0)
            if val_mask.sum() > 0:
                val_ce = ce(logits_val[val_mask], y[val_mask]) / val_mask.sum()
                n_val = int(val_mask.sum().item())
            else:
                val_ce = torch.tensor(0.0, device=device)
                n_val = 1
            val_loss = acfg.lambda_recon * loss_recon_val + val_ce
    train_avg = total_train / max(total_nodes, 1)
    return train_avg, float(val_loss.item())


def train_with_early_stopping(
    model: SupDAEGraphNodeFeat,
    dl: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    history: List[float],
    logger: Optional[ContinuousLogger] = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for epoch in range(1, acfg.max_epochs + 1):
        model.train()
        train_loss, val_loss = train_val_one_pass(model, dl, opt, acfg, device)
        history.append(val_loss)

        improved = val_loss < best_val - acfg.delta
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1

        msg = (
            f"  Epoch {epoch:03d}/{acfg.max_epochs} | "
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


def adp_search(
    model: SupDAEGraphNodeFeat,
    dl: DataLoader,
    in_dim: int,
    num_classes: int,
    acfg: ADPConfig,
    device: torch.device,
    logger: ContinuousLogger,
    results_dir: Path,
    log_loss: bool = False,
    log_neurons: bool = False,
) -> Tuple[float, SupDAEGraphNodeFeat, int, int]:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.log_console(f"[ADP] Mode={acfg.adp_mode}")

    val_history: List[float] = []
    # (neurons, val_loss)
    improvements: List[Tuple[int, float]] = []

    # 1. Initial training to get a baseline
    global_best_val, global_best_state = train_with_early_stopping(
        model, dl, acfg, device, val_history, logger=logger
    )
    global_best_snap = snapshot_arch_and_state(model, global_best_state)
    logger.log_console(f"[ADP] Baseline: val={global_best_val:.6f}, w={model.width}, d={model.depth}")

    # Helper predicates
    def can_widen(m: SupDAEGraphNodeFeat) -> bool:
        return (
            sup_dae_total_neurons(m.width + acfg.ex_k, m.depth, num_classes) <= acfg.max_neurons
            and m.width < acfg.max_width
        )

    def can_deepen(m: SupDAEGraphNodeFeat) -> bool:
        return (
            sup_dae_total_neurons(m.width, m.depth + 1, num_classes) <= acfg.max_neurons
            and m.depth < acfg.max_depth
        )

    # --- Search Primitives (No Rollback per step) ---

    def optimize_width_at_fixed_depth(model_curr: SupDAEGraphNodeFeat) -> Tuple[float, Dict[str, torch.Tensor], int]:
        """
        Expands width from model_curr until patience exhausted.
        Returns (best_val, best_state, best_width) found during this local search.
        Reverts model_curr to that local best at the end.
        """
        best_v, best_s = train_with_early_stopping(model_curr, dl, acfg, device, val_history, logger)
        best_w = model_curr.width
        best_local_snap = snapshot_arch_and_state(model_curr, best_s)

        fail_count = 0
        while fail_count < acfg.trials_width:
            if not can_widen(model_curr):
                break
            
            # Always expand from CURRENT architecture (No Rollback on fail)
            wider = expand_width(model_curr, in_dim, num_classes, acfg.ex_k, acfg.max_width, device)
            if wider is None: break
            model_curr = wider
            
            val, state = train_with_early_stopping(model_curr, dl, acfg, device, val_history, logger)
            
            if val < best_v - acfg.delta:
                best_v = val
                best_s = state
                best_w = model_curr.width
                best_local_snap = snapshot_arch_and_state(model_curr, state)
                fail_count = 0
                
                neurons = sup_dae_total_neurons(model_curr.width, model_curr.depth, num_classes)
                improvements.append((neurons, val))
                logger.log_console(f"    [Width Inner] Improved! w={model_curr.width} val={val:.6f}")
            else:
                fail_count += 1
                # logger.log_console(f"    [Width Inner] Fail {fail_count}/{acfg.trials_width} w={model_curr.width} val={val:.6f}")

        restore_arch_and_state(best_local_snap, device)
        # We must return state/val so caller can update global best if needed
        return best_v, best_s, best_w

    def optimize_depth_at_fixed_width(model_curr: SupDAEGraphNodeFeat) -> Tuple[float, Dict[str, torch.Tensor], int]:
        """
        Expands depth from model_curr until patience exhausted.
        Returns (best_val, best_state, best_depth).
        Reverts model_curr to that local best at the end.
        """
        best_v, best_s = train_with_early_stopping(model_curr, dl, acfg, device, val_history, logger)
        best_d = model_curr.depth
        best_local_snap = snapshot_arch_and_state(model_curr, best_s)

        fail_count = 0
        while fail_count < acfg.trials_depth:
            if not can_deepen(model_curr):
                break
            
            # Always expand from CURRENT architecture
            deeper = expand_depth(model_curr, in_dim, num_classes, acfg.max_depth, device)
            if deeper is None: break
            model_curr = deeper
            
            val, state = train_with_early_stopping(model_curr, dl, acfg, device, val_history, logger)
            
            if val < best_v - acfg.delta:
                best_v = val
                best_s = state
                best_d = model_curr.depth
                best_local_snap = snapshot_arch_and_state(model_curr, state)
                fail_count = 0
                
                neurons = sup_dae_total_neurons(model_curr.width, model_curr.depth, num_classes)
                improvements.append((neurons, val))
                logger.log_console(f"    [Depth Inner] Improved! d={model_curr.depth} val={val:.6f}")
            else:
                fail_count += 1
                # logger.log_console(f"    [Depth Inner] Fail {fail_count}/{acfg.trials_depth} d={model_curr.depth} val={val:.6f}")

        restore_arch_and_state(best_local_snap, device)
        return best_v, best_s, best_d

    # --- Mode Implementations ---

    mode = acfg.adp_mode

    # 1. Depth Only
    if mode in ("depth_only", "depth"):
        # We rely on the helper which tracks best internally but walks forward
        optimize_depth_at_fixed_width(model)
        # The helper restores the best local model at the end, so 'model' is now best
        global_best_val, global_best_state = train_with_early_stopping(model, dl, acfg, device, [], None)
        global_best_snap = snapshot_arch_and_state(model, global_best_state)

    # 2. Width Only
    elif mode in ("width_only", "width"):
        optimize_width_at_fixed_depth(model)
        global_best_val, global_best_state = train_with_early_stopping(model, dl, acfg, device, [], None)
        global_best_snap = snapshot_arch_and_state(model, global_best_state)

    # 3. Depth-Outer / Width-Inner (Mapped from 'depth_to_width')
    elif mode == "depth_to_width":
        # First inner search at starting depth
        optimize_width_at_fixed_depth(model)
        
        # Update global best
        b_val, b_state = train_with_early_stopping(model, dl, acfg, device, [], None)
        if b_val < global_best_val:
            global_best_val = b_val
            global_best_snap = snapshot_arch_and_state(model, b_state)
        
        fail_count = 0
        while fail_count < acfg.trials_depth and can_deepen(model):
            # Outer: Expand depth
            deeper = expand_depth(model, in_dim, num_classes, acfg.max_depth, device)
            if deeper is None: break
            model = deeper
            
            # Inner: Optimize width at this new depth
            val_d, state_d, width_d = optimize_width_at_fixed_depth(model)
            # note: optimize_width_at_fixed_depth leaves model at best width for this depth
            
            if val_d < global_best_val - acfg.delta:
                global_best_val = val_d
                global_best_snap = snapshot_arch_and_state(model, state_d)
                fail_count = 0
                logger.log_console(f"[Outer Depth] New Global Best: val={val_d:.6f} d={model.depth} w={model.width}")
            else:
                fail_count += 1

    # 4. Width-Outer / Depth-Inner (Mapped from 'width_to_depth')
    elif mode == "width_to_depth":
        # First inner search at starting width
        optimize_depth_at_fixed_width(model)
        
        b_val, b_state = train_with_early_stopping(model, dl, acfg, device, [], None)
        if b_val < global_best_val:
            global_best_val = b_val
            global_best_snap = snapshot_arch_and_state(model, b_state)

        fail_count = 0
        while fail_count < acfg.trials_width and can_widen(model):
            # Outer: Expand width
            wider = expand_width(model, in_dim, num_classes, acfg.ex_k, acfg.max_width, device)
            if wider is None: break
            model = wider
            
            # Inner: Optimize depth at this new width
            val_w, state_w, depth_w = optimize_depth_at_fixed_width(model)
            
            if val_w < global_best_val - acfg.delta:
                global_best_val = val_w
                global_best_snap = snapshot_arch_and_state(model, state_w)
                fail_count = 0
                logger.log_console(f"[Outer Width] New Global Best: val={val_w:.6f} w={model.width} d={model.depth}")
            else:
                fail_count += 1

    # 5/6. Alternating Phases
    elif mode in ("alt_width", "alt_depth"):
        # Ensure we start from global best
        restore_arch_and_state(global_best_snap, device)
        
        phase = "width" if mode == "alt_width" else "depth"
        sat_w, sat_d = False, False
        
        while not (sat_w and sat_d):
            improved_in_phase = False
            start_val = global_best_val
            
            if phase == "width":
                logger.log_console("  [Phase] Entering Width Phase")
                optimize_width_at_fixed_depth(model)
                # Check if we improved over start of phase
                now_val, now_state = train_with_early_stopping(model, dl, acfg, device, [], None)
                if now_val < global_best_val - acfg.delta:
                    global_best_val = now_val
                    global_best_snap = snapshot_arch_and_state(model, now_state)
                    improved_in_phase = True
                
                sat_w = not improved_in_phase
                restore_arch_and_state(global_best_snap, device) # Revert to best for next phase
                phase = "depth"
                
            else: # depth
                logger.log_console("  [Phase] Entering Depth Phase")
                optimize_depth_at_fixed_width(model)
                now_val, now_state = train_with_early_stopping(model, dl, acfg, device, [], None)
                if now_val < global_best_val - acfg.delta:
                    global_best_val = now_val
                    global_best_snap = snapshot_arch_and_state(model, now_state)
                    improved_in_phase = True
                
                sat_d = not improved_in_phase
                restore_arch_and_state(global_best_snap, device)
                phase = "width"

    else:
        logger.log_console(f"[WARN] Unknown adp_mode={mode}, skipping search.")

    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title="SupDAEGraphNodeFeat")
    if log_neurons and improvements:
        ns = [n for n, _ in improvements]
        vs = [v for _, v in improvements]
        plot_loss_vs_neurons(ns, vs, results_dir / "loss_vs_neurons.png", title="SupDAEGraphNodeFeat")

    final_model = restore_arch_and_state(global_best_snap, device)
    return global_best_val, final_model, final_model.width, final_model.depth


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="ADP supervised graph node-feature DAE encoder + node classifier")
    p.add_argument("--graph-path", type=str, required=True)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"],
    )
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-width", type=int, default=256)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--results-dir", type=str, default="results_adp_dae_graph_nodefeat_sup")
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds, in_dim, num_classes = build_dataset(args.graph_path, device)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = SupDAEGraphNodeFeat(in_dim=in_dim, num_classes=num_classes, width=args.width, depth=args.depth).to(device)

    acfg = ADPConfig(
        adp_mode=args.adp_mode,
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
        grad_clip=args.grad_clip,
        max_epochs=args.max_epochs,
        noise_std=args.noise_std,
        lambda_recon=args.lambda_recon,
    )

    results_dir = Path(args.results_dir)
    logger = ContinuousLogger(results_dir, "sup_dae_graph_nodefeat", args.adp_mode)

    best_val, best_model, best_w, best_d = adp_search(
        model,
        dl,
        in_dim=in_dim,
        num_classes=num_classes,
        acfg=acfg,
        device=device,
        logger=logger,
        results_dir=results_dir,
        log_loss=args.plot_loss,
        log_neurons=args.plot_neurons,
    )

    logger.log_console(f"[DONE] Best val={best_val:.6f}, width={best_w}, depth={best_d}")
    logger.close()
