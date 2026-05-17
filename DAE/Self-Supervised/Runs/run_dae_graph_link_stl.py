import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from ..Models.dae_graph_link_stl import DAEGraphLink, graph_link_dae_total_neurons


class AdjacencyDataset(Dataset):
    """
    Dataset of one or more adjacency matrices.

    Expected .pt at --graph-path:
        {"adj": Tensor[G, N, N]} or Tensor[N, N] or (adj,) tuple.
    """

    def __init__(self, adj: torch.Tensor):
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        assert adj.dim() == 3, "adj must be (G,N,N) or (N,N)"
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
        raise ValueError("Graph file must contain adjacency as key 'adj' or be a Tensor/tuple with adjacency.")
    return AdjacencyDataset(adj.float())


def build_dataloaders(
    graph_path: str,
    batch_size: int,
    num_workers: int,
    val_frac: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)
    ds = load_adj_dataset(graph_path)

    if len(ds) == 1 or val_frac <= 0.0:
        train_ds = val_ds = ds
    else:
        val_size = max(1, int(len(ds) * val_frac))
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def drop_edges(adj: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0.0:
        return adj
    mask = torch.rand_like(adj) >= drop_prob
    return adj * mask


def train_one_epoch(
    model: DAEGraphLink,
    loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    drop_prob: float,
) -> float:
    model.train()
    mse = nn.MSELoss()
    total, n = 0.0, 0
    for adj in loader:
        adj = adj.to(device)
        if adj.dim() == 3:
            adj = adj.squeeze(0)

        adj_noisy = drop_edges(adj, drop_prob)

        opt.zero_grad(set_to_none=True)
        adj_rec, _ = model(adj_noisy)
        loss = mse(adj_rec, adj)
        loss.backward()
        opt.step()

        total += float(loss.item()) * adj.size(0)
        n += adj.size(0)
    return total / max(n, 1)


def eval_epoch(
    model: DAEGraphLink,
    loader: DataLoader,
    device: torch.device,
    drop_prob: float,
) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    with torch.no_grad():
        for adj in loader:
            adj = adj.to(device)
            if adj.dim() == 3:
                adj = adj.squeeze(0)
            adj_noisy = drop_edges(adj, drop_prob)
            adj_rec, _ = model(adj_noisy)
            total += float(mse(adj_rec, adj).item())
            n += adj.size(0)
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Graph link-structure DAE STL")
    p.add_argument("--graph-path", type=str, required=True, help="Path to .pt file with adjacency")
    p.add_argument("--out-dir", type=str, default="./Runs/dae_graph_link_stl")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--n-nodes", type=int, default=None, help="Override number of nodes; otherwise inferred")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--drop-prob", type=float, default=0.1)

    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        graph_path=args.graph_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    adj0 = next(iter(train_loader))
    if adj0.dim() == 3:
        adj0 = adj0.squeeze(0)
    n_nodes = args.n_nodes if args.n_nodes is not None else adj0.size(-1)

    model = DAEGraphLink(n_nodes=n_nodes, width=args.width, depth=args.depth).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "training_log.txt"
    stats_path = out_dir / "training_stats.csv"
    ckpt_path = out_dir / "best.pt"

    log_f = log_path.open("w", encoding="utf-8")
    stats_f = stats_path.open("w", newline="", encoding="utf-8")
    stats_writer = csv.writer(stats_f)
    stats_writer.writerow(
        ["epoch", "width", "depth", "neurons", "train_loss", "val_loss", "best_val", "best_epoch"]
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    neurons = graph_link_dae_total_neurons(args.width, args.depth)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, opt, device, drop_prob=args.drop_prob)
            val_loss = eval_epoch(model, val_loader, device, drop_prob=args.drop_prob)

            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
            else:
                epochs_no_improve += 1

            msg = (
                f"Epoch {epoch:03d} | train={train_loss:.6f} | "
                f"val={val_loss:.6f} | best_val={best_val:.6f} @ {best_epoch}"
            )
            print(msg)
            log_f.write(msg + "\n")

            stats_writer.writerow([epoch, args.width, args.depth, neurons, train_loss, val_loss, best_val, best_epoch])
            stats_f.flush()

            if epochs_no_improve >= args.patience:
                stop_msg = f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)"
                print(stop_msg)
                log_f.write(stop_msg + "\n")
                break
    finally:
        log_f.flush()
        stats_f.flush()

    report = {
        "graph_path": args.graph_path,
        "n_nodes": n_nodes,
        "width": args.width,
        "depth": args.depth,
        "neurons_metric": neurons,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "drop_prob": args.drop_prob,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log_f.write("\n" + json.dumps(report, indent=2) + "\n")
    log_f.close()
    stats_f.close()


if __name__ == "__main__":
    main()

