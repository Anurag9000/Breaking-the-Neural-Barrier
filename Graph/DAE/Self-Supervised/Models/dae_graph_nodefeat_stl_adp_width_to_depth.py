import copy
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.append(str(Path(__file__).resolve().parents[4]))
from utils.adp_logging import ContinuousLogger  # type: ignore
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

from .dae_graph_nodefeat_stl import DAEGraphNodeFeat, graph_node_dae_total_neurons


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
    grad_clip: float = 1.0
    max_epochs: int = 300
    noise_std: float = 0.1


class SingleGraphDataset(Dataset):
    """
    Dataset wrapper for one or more graphs with node features and adjacency.
    """

    def __init__(self, x: torch.Tensor, adj: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        assert x.dim() == 3 and adj.dim() == 3, "x must be (G,N,F), adj (G,N,N)"
        assert x.size(0) == adj.size(0)
        self.x = x
        self.adj = adj

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.adj[idx]


def load_graph_dataset(graph_path: str) -> SingleGraphDataset:
    obj = torch.load(graph_path, map_location="cpu")
    if isinstance(obj, dict):
        x = obj.get("x")
        adj = obj.get("adj")
    elif isinstance(obj, (tuple, list)) and len(obj) >= 2:
        x, adj = obj[0], obj[1]
    else:
        raise ValueError("Graph file must be dict with 'x' and 'adj' or tuple (x, adj).")
    if x is None or adj is None:
        raise ValueError("Graph file missing 'x' or 'adj'.")
    return SingleGraphDataset(x.float(), adj.float())


def make_loaders(
    graph_path: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(1337)
    ds = load_graph_dataset(graph_path)
    if len(ds) == 1 or val_split <= 0.0:
        train_ds = val_ds = ds
    else:
        val_size = max(1, int(len(ds) * val_split))
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


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
    model: DAEGraphNodeFeat,
    width: int,
    depth: int,
    device: torch.device,
) -> DAEGraphNodeFeat:
    new_model = DAEGraphNodeFeat(in_dim=model.in_dim, width=width, depth=depth).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(
    model: DAEGraphNodeFeat,
    ex_k: int,
    max_width: int,
    device: torch.device,
) -> Optional[DAEGraphNodeFeat]:
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return None
    return rebuild_model(model, new_w, model.depth, device)


def expand_depth(
    model: DAEGraphNodeFeat,
    max_depth: int,
    device: torch.device,
) -> Optional[DAEGraphNodeFeat]:
    if model.depth >= max_depth:
        return None
    return rebuild_model(model, model.width, model.depth + 1, device)


def snapshot_arch_and_state(model: DAEGraphNodeFeat, state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    st = state if state is not None else model.state_dict()
    return {
        "in_dim": model.in_dim,
        "width": model.width,
        "depth": model.depth,
        "state": copy.deepcopy(st),
    }


def restore_arch_and_state(snap: Dict[str, Any], device: torch.device) -> DAEGraphNodeFeat:
    mdl = DAEGraphNodeFeat(
        in_dim=snap["in_dim"],
        width=snap["width"],
        depth=snap["depth"],
    ).to(device)
    mdl.load_state_dict(snap["state"], strict=False)
    return mdl


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def train_with_early_stopping(
    model: DAEGraphNodeFeat,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    history: List[float],
    logger: Optional[ContinuousLogger] = None,
    verbose: bool = True,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    mse = nn.MSELoss()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for epoch in range(1, acfg.max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, adj in dl_train:
            xb = xb.to(device)
            adj = adj.to(device)
            if xb.dim() == 3:
                xb = xb.squeeze(0)
            if adj.dim() == 3:
                adj = adj.squeeze(0)

            xb_noisy = add_gaussian_noise(xb, acfg.noise_std)

            opt.zero_grad(set_to_none=True)
            xb_rec, _ = model(xb_noisy, adj)
            loss = mse(xb_rec, xb)
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
            for xb, adj in dl_val:
                xb = xb.to(device)
                adj = adj.to(device)
                if xb.dim() == 3:
                    xb = xb.squeeze(0)
                if adj.dim() == 3:
                    adj = adj.squeeze(0)
                xb_noisy = add_gaussian_noise(xb, acfg.noise_std)
                xb_rec, _ = model(xb_noisy, adj)
                total += float(mse(xb_rec, xb).item())
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
            f"  Epoch {epoch:03d}/{acfg.max_epochs} | "
            f"Train={train_loss:.6f} | Val={val_loss:.6f} | "
            f"Best={best_val:.6f} | ES={es_counter}/{acfg.patience}"
        )
        if logger:
            logger.log_console(msg)
        elif verbose:
            print(msg)

        if es_counter >= acfg.patience:
            if logger:
                logger.log_console(f"  Early stopping at epoch {epoch}")
            elif verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    return best_val, best_state


def adp_search(
    model: DAEGraphNodeFeat,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    logger: ContinuousLogger,
    results_dir: Path,
    log_loss: bool = False,
    log_neurons: bool = False,
) -> Tuple[float, DAEGraphNodeFeat, int, int]:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.log_console(f"[ADP] Mode={acfg.adp_mode}")

    val_history: List[float] = []
    improvements: List[Tuple[int, float]] = []

    global_best_val, global_best_state = train_with_early_stopping(
        model, dl_train, dl_val, acfg, device, val_history, logger=logger
    )
    global_best_snap = snapshot_arch_and_state(model, global_best_state)

    def can_widen(m: DAEGraphNodeFeat) -> bool:
        return graph_node_dae_total_neurons(m.width + acfg.ex_k, m.depth) <= acfg.max_neurons and m.width < acfg.max_width

    def can_deepen(m: DAEGraphNodeFeat) -> bool:
        return graph_node_dae_total_neurons(m.width, m.depth + 1) <= acfg.max_neurons and m.depth < acfg.max_depth

    def optimize_width_at_fixed_depth(
        snap: Dict[str, Any],
        best_val: float,
    ) -> Tuple[Dict[str, Any], float]:
        local_best_val = best_val
        local_best_snap = snap
        fail = 0
        while fail < acfg.trials_width:
            curr = restore_arch_and_state(local_best_snap, device)
            if not can_widen(curr):
                break
            wider = expand_width(curr, acfg.ex_k, acfg.max_width, device)
            if wider is None:
                break
            v, s = train_with_early_stopping(wider, dl_train, dl_val, acfg, device, val_history, logger=logger)
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(wider, s)
                fail = 0
                improvements.append((graph_node_dae_total_neurons(wider.width, wider.depth), v))
                logger.log_console(f"[WIDTH OPT] ✓ New best width={wider.width}, val={v:.6f}")
            else:
                fail += 1
                logger.log_console("[WIDTH OPT] ✗ No improvement")
        return local_best_snap, local_best_val

    def optimize_depth_at_fixed_width(
        snap: Dict[str, Any],
        best_val: float,
    ) -> Tuple[Dict[str, Any], float]:
        local_best_val = best_val
        local_best_snap = snap
        fail = 0
        while fail < acfg.trials_depth:
            curr = restore_arch_and_state(local_best_snap, device)
            if not can_deepen(curr):
                break
            deeper = expand_depth(curr, acfg.max_depth, device)
            if deeper is None:
                break
            v, s = train_with_early_stopping(deeper, dl_train, dl_val, acfg, device, val_history, logger=logger)
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(deeper, s)
                fail = 0
                improvements.append((graph_node_dae_total_neurons(deeper.width, deeper.depth), v))
                logger.log_console(f"[DEPTH OPT] ✓ New best depth={deeper.depth}, val={v:.6f}")
            else:
                fail += 1
                logger.log_console("[DEPTH OPT] ✗ No improvement")
        return local_best_snap, local_best_val

    mode = acfg.adp_mode

    if mode in ["width_only", "width"]:
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
    elif mode in ["depth_only", "depth"]:
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
    elif mode == "width_to_depth":
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
        fail = 0
        while fail < acfg.trials_width:
            tmp = restore_arch_and_state(global_best_snap, device)
            if not can_widen(tmp):
                break
            wider = expand_width(tmp, acfg.ex_k, acfg.max_width, device)
            wider_snap = snapshot_arch_and_state(wider, wider.state_dict())
            wider_snap, val = optimize_depth_at_fixed_width(wider_snap, global_best_val)
            if val < global_best_val - acfg.delta:
                global_best_val = val
                global_best_snap = wider_snap
                fail = 0
            else:
                fail += 1
    elif mode == "depth_to_width":
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
        fail = 0
        while fail < acfg.trials_depth:
            tmp = restore_arch_and_state(global_best_snap, device)
            if not can_deepen(tmp):
                break
            deeper = expand_depth(tmp, acfg.max_depth, device)
            deeper_snap = snapshot_arch_and_state(deeper, deeper.state_dict())
            deeper_snap, val = optimize_width_at_fixed_depth(deeper_snap, global_best_val)
            if val < global_best_val - acfg.delta:
                global_best_val = val
                global_best_snap = deeper_snap
                fail = 0
            else:
                fail += 1
    elif mode in ["alt_width", "alt_depth"]:
        phase = "width" if mode == "alt_width" else "depth"
        sat_w = sat_d = False
        while not (sat_w and sat_d):
            improved = False
            if phase == "width":
                snap, val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved = True
                sat_w = not improved
                phase = "depth"
            else:
                snap, val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved = True
                sat_d = not improved
                phase = "width"
    else:
        logger.log_console(f"[WARN] Unknown adp_mode={mode}, skipping search.")

    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title="DAEGraphNodeFeat")
    if log_neurons and improvements:
        ns = [n for n, _ in improvements]
        vs = [v for _, v in improvements]
        plot_loss_vs_neurons(ns, vs, results_dir / "loss_vs_neurons.png", title="DAEGraphNodeFeat")

    final_model = restore_arch_and_state(global_best_snap, device)
    return global_best_val, final_model, final_model.width, final_model.depth


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="ADP graph node-feature DAE")
    p.add_argument("--graph-path", type=str, required=True, help="Path to .pt file with 'x' and 'adj'.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-split", type=float, default=0.2)

    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)

    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"],
    )
    p.add_argument("--max-epochs", type=int, default=300)
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
    p.add_argument("--noise-std", type=float, default=0.1)

    p.add_argument("--results-dir", type=str, default="results_adp_dae_graph_nodefeat")
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_train, dl_val = make_loaders(
        graph_path=args.graph_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    # Infer in_dim from first batch
    xb0, adj0 = next(iter(dl_train))
    if xb0.dim() == 3:
        xb0 = xb0.squeeze(0)
    in_dim = xb0.size(-1)

    model = DAEGraphNodeFeat(in_dim=in_dim, width=args.width, depth=args.depth).to(device)

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
    )

    results_dir = Path(args.results_dir)

    logger = ContinuousLogger(results_dir, "dae_graph_nodefeat", args.adp_mode)

    best_val, best_model, best_w, best_d = adp_search(
        model,
        dl_train,
        dl_val,
        acfg,
        device,
        logger=logger,
        results_dir=results_dir,
        log_loss=args.plot_loss,
        log_neurons=args.plot_neurons,
    )

    logger.log_console(f"[DONE] Best val={best_val:.6f}, width={best_w}, depth={best_d}")
    logger.close()
