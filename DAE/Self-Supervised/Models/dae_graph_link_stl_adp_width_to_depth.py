import copy
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger  # type: ignore
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

from .dae_graph_link_stl import DAEGraphLink, graph_link_dae_total_neurons


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
    drop_prob: float = 0.1


class AdjacencyDataset(Dataset):
    def __init__(self, adj: torch.Tensor):
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        assert adj.dim() == 3
        self.adj = adj

    def __len__(self) -> int:
        return self.adj.size(0)

    def __getitem__(self, idx: int):
        return self.adj[idx]


def load_adj_dataset(graph_path: str) -> AdjacencyDataset:
    obj = torch.load(graph_path, map_location="cpu")
    if isinstance(obj, dict):
        adj = obj.get("adj")
    elif isinstance(obj, (tuple, list)) and len(obj) >= 1:
        adj = obj[0]
    else:
        adj = obj
    if adj is None:
        raise ValueError("Graph file must contain adjacency as 'adj' or be a Tensor/tuple.")
    return AdjacencyDataset(adj.float())


def make_loaders(
    graph_path: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(1337)
    ds = load_adj_dataset(graph_path)
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


def rebuild_model(model: DAEGraphLink, width: int, depth: int, device: torch.device) -> DAEGraphLink:
    new_model = DAEGraphLink(n_nodes=model.n_nodes, width=width, depth=depth).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(model: DAEGraphLink, ex_k: int, max_width: int, device: torch.device) -> Optional[DAEGraphLink]:
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return None
    return rebuild_model(model, new_w, model.depth, device)


def expand_depth(model: DAEGraphLink, max_depth: int, device: torch.device) -> Optional[DAEGraphLink]:
    if model.depth >= max_depth:
        return None
    return rebuild_model(model, model.width, model.depth + 1, device)


def snapshot_arch_and_state(model: DAEGraphLink, state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    st = state if state is not None else model.state_dict()
    return {
        "n_nodes": model.n_nodes,
        "width": model.width,
        "depth": model.depth,
        "state": copy.deepcopy(st),
    }


def restore_arch_and_state(snap: Dict[str, Any], device: torch.device) -> DAEGraphLink:
    mdl = DAEGraphLink(
        n_nodes=snap["n_nodes"],
        width=snap["width"],
        depth=snap["depth"],
    ).to(device)
    mdl.load_state_dict(snap["state"], strict=False)
    return mdl


def drop_edges(adj: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0.0:
        return adj
    mask = torch.rand_like(adj) >= drop_prob
    return adj * mask


def train_with_early_stopping(
    model: DAEGraphLink,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    history: List[float],
    logger: Optional[ContinuousLogger] = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    mse = nn.MSELoss()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for epoch in range(1, acfg.max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for adj in dl_train:
            adj = adj.to(device)
            if adj.dim() == 3:
                adj = adj.squeeze(0)
            adj_noisy = drop_edges(adj, acfg.drop_prob)

            opt.zero_grad(set_to_none=True)
            adj_rec, _ = model(adj_noisy)
            loss = mse(adj_rec, adj)
            loss.backward()
            if acfg.grad_clip is not None and acfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()

            total += float(loss.item()) * adj.size(0)
            n += adj.size(0)
        train_loss = total / max(n, 1)

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for adj in dl_val:
                adj = adj.to(device)
                if adj.dim() == 3:
                    adj = adj.squeeze(0)
                adj_noisy = drop_edges(adj, acfg.drop_prob)
                adj_rec, _ = model(adj_noisy)
                total += float(mse(adj_rec, adj).item())
                n += adj.size(0)
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
    model: DAEGraphLink,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    logger: ContinuousLogger,
    results_dir: Path,
    log_loss: bool = False,
    log_neurons: bool = False,
) -> Tuple[float, DAEGraphLink, int, int]:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.log_console(f"[ADP] Mode={acfg.adp_mode}")

    val_history: List[float] = []
    improvements: List[Tuple[int, float]] = []

    global_best_val, global_best_state = train_with_early_stopping(
        model, dl_train, dl_val, acfg, device, val_history, logger=logger
    )
    global_best_snap = snapshot_arch_and_state(model, global_best_state)

    def can_widen(m: DAEGraphLink) -> bool:
        return (
            graph_link_dae_total_neurons(m.width + acfg.ex_k, m.depth) <= acfg.max_neurons
            and m.width < acfg.max_width
        )

    def can_deepen(m: DAEGraphLink) -> bool:
        return (
            graph_link_dae_total_neurons(m.width, m.depth + 1) <= acfg.max_neurons
            and m.depth < acfg.max_depth
        )

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
                improvements.append((graph_link_dae_total_neurons(wider.width, wider.depth), v))
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
                improvements.append((graph_link_dae_total_neurons(deeper.width, deeper.depth), v))
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
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title="DAEGraphLink")
    if log_neurons and improvements:
        ns = [n for n, _ in improvements]
        vs = [v for _, v in improvements]
        plot_loss_vs_neurons(ns, vs, results_dir / "loss_vs_neurons.png", title="DAEGraphLink")

    final_model = restore_arch_and_state(global_best_snap, device)
    return global_best_val, final_model, final_model.width, final_model.depth


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="ADP Graph link-structure DAE")
    p.add_argument("--graph-path", type=str, required=True, help="Path to .pt adjacency file")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-split", type=float, default=0.2)

    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)

    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"],
    )
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=16)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--drop-prob", type=float, default=0.1)

    p.add_argument("--results-dir", type=str, default="results_adp_dae_graph_link")
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

    adj0 = next(iter(dl_train))
    if adj0.dim() == 3:
        adj0 = adj0.squeeze(0)
    n_nodes = adj0.size(-1)

    model = DAEGraphLink(n_nodes=n_nodes, width=args.width, depth=args.depth).to(device)

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
        drop_prob=args.drop_prob,
    )

    results_dir = Path(args.results_dir)
    logger = ContinuousLogger(results_dir, "dae_graph_link", args.adp_mode)

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


if __name__ == "__main__":
    main()

