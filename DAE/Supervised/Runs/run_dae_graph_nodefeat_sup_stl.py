import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ..Models.dae_graph_nodefeat_sup_stl import SupDAEGraphNodeFeat, sup_dae_total_neurons


class LabeledGraphDataset(Dataset):
    """
    Single graph with node features, adjacency, labels, and masks.

    Expected .pt format at --graph-path:
      {
        "x": Tensor[N, F],
        "adj": Tensor[N, N],
        "y": Tensor[N] (long),
        "train_mask": BoolTensor[N],
        "val_mask":   BoolTensor[N],
        "test_mask":  BoolTensor[N],
      }
    If masks are missing, we create a 50/25/25 split.
    """

    def __init__(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        y: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
    ) -> None:
        self.x = x
        self.adj = adj
        self.y = y
        self.train_mask = train_mask.bool()
        self.val_mask = val_mask.bool()
        self.test_mask = test_mask.bool()

    def __len__(self) -> int:
        # Single-graph dataset: one sample containing the whole graph.
        return 1

    def __getitem__(self, idx: int):
        return self.x, self.adj, self.y, self.train_mask, self.val_mask, self.test_mask


def build_dataset(graph_path: str, device: torch.device) -> Tuple[LabeledGraphDataset, int, int]:
    obj = torch.load(graph_path, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError("Graph file must be a dict with x, adj, y and optional masks.")

    x = obj["x"].float()
    adj = obj["adj"].float()
    y = obj["y"].long()
    n = x.size(0)
    num_classes = int(y.max().item() + 1)

    train_mask = obj.get("train_mask")
    val_mask = obj.get("val_mask")
    test_mask = obj.get("test_mask")
    if train_mask is None or val_mask is None or test_mask is None:
        idx = torch.randperm(n, device=device)
        n_train = max(1, int(0.5 * n))
        n_val = max(1, int(0.25 * n))
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        train_mask = torch.zeros(n, dtype=torch.bool, device=device)
        val_mask = torch.zeros(n, dtype=torch.bool, device=device)
        test_mask = torch.zeros(n, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
    else:
        train_mask = train_mask.to(device).bool()
        val_mask = val_mask.to(device).bool()
        test_mask = test_mask.to(device).bool()

    ds = LabeledGraphDataset(x.to(device), adj.to(device), y.to(device), train_mask, val_mask, test_mask)
    in_dim = x.size(1)
    return ds, in_dim, num_classes


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def train_one_epoch(
    model: SupDAEGraphNodeFeat,
    loader: DataLoader,
    opt: optim.Optimizer,
    noise_std: float,
    lambda_recon: float,
) -> Tuple[float, float]:
    model.train()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    total_loss, total_cls, total_nodes = 0.0, 0.0, 0

    for x, adj, y, train_mask, _, _ in loader:
        x = x.squeeze(0)
        adj = adj.squeeze(0)
        y = y.squeeze(0)
        train_mask = train_mask.squeeze(0)

        x_noisy = add_gaussian_noise(x, noise_std)

        opt.zero_grad(set_to_none=True)
        x_rec, logits = model(x_noisy, adj)
        loss_recon = mse(x_rec, x)
        if train_mask.sum() > 0:
            loss_cls = ce(logits[train_mask], y[train_mask])
        else:
            loss_cls = torch.tensor(0.0, device=x.device)
        loss = lambda_recon * loss_recon + loss_cls
        loss.backward()
        opt.step()

        n = max(int(train_mask.sum().item()), 1)
        total_loss += float(loss.item()) * n
        total_cls += float(loss_cls.item()) * n
        total_nodes += n

    return total_loss / max(total_nodes, 1), total_cls / max(total_nodes, 1)


def eval_epoch(
    model: SupDAEGraphNodeFeat,
    loader: DataLoader,
    noise_std: float,
    lambda_recon: float,
) -> Tuple[float, float, float]:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_cls, total_nodes, correct = 0.0, 0.0, 0, 0

    with torch.no_grad():
        for x, adj, y, _, val_mask, _ in loader:
            x = x.squeeze(0)
            adj = adj.squeeze(0)
            y = y.squeeze(0)
            val_mask = val_mask.squeeze(0)

            x_noisy = add_gaussian_noise(x, noise_std)
            x_rec, logits = model(x_noisy, adj)

            loss_recon = mse(x_rec, x) / x.size(0)
            if val_mask.sum() > 0:
                loss_cls = ce(logits[val_mask], y[val_mask]) / val_mask.sum()
                preds = logits[val_mask].argmax(dim=1)
                correct += int((preds == y[val_mask]).sum().item())
                n = int(val_mask.sum().item())
            else:
                loss_cls = torch.tensor(0.0, device=x.device)
                n = 1

            loss = lambda_recon * loss_recon + loss_cls
            total_loss += float(loss.item()) * n
            total_cls += float(loss_cls.item()) * n
            total_nodes += n

    acc = correct / max(total_nodes, 1)
    return total_loss / max(total_nodes, 1), total_cls / max(total_nodes, 1), acc


def eval_test_accuracy(
    model: SupDAEGraphNodeFeat,
    loader: DataLoader,
    noise_std: float,
) -> float:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct, total = 0, 0
    with torch.no_grad():
        for x, adj, y, _, _, test_mask in loader:
            x = x.squeeze(0)
            adj = adj.squeeze(0)
            y = y.squeeze(0)
            test_mask = test_mask.squeeze(0)

            x_noisy = add_gaussian_noise(x, noise_std)
            _, logits = model(x_noisy, adj)
            if test_mask.sum() == 0:
                continue
            preds = logits[test_mask].argmax(dim=1)
            correct += int((preds == y[test_mask]).sum().item())
            total += int(test_mask.sum().item())
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Supervised graph node-feature DAE encoder + node classifier")
    p.add_argument("--graph-path", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="./runs/dae_graph_nodefeat_sup")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--lambda-recon", type=float, default=1.0)

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds, in_dim, num_classes = build_dataset(args.graph_path, device)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = SupDAEGraphNodeFeat(in_dim=in_dim, num_classes=num_classes, width=args.width, depth=args.depth).to(device)
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
        ["epoch", "width", "depth", "neurons", "train_loss", "train_cls", "val_loss", "val_cls", "val_acc", "best_val", "best_epoch"]
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    neurons = sup_dae_total_neurons(args.width, args.depth, num_classes)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_cls = train_one_epoch(
                model, loader, opt, noise_std=args.noise_std, lambda_recon=args.lambda_recon
            )
            val_loss, val_cls, val_acc = eval_epoch(
                model, loader, noise_std=args.noise_std, lambda_recon=args.lambda_recon
            )

            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "args": vars(args)},
                    ckpt_path,
                )
            else:
                epochs_no_improve += 1

            msg = (
                f"Epoch {epoch:03d} | train={train_loss:.6f} (cls={train_cls:.6f}) | "
                f"val={val_loss:.6f} (cls={val_cls:.6f}, acc={val_acc:.4f}) | "
                f"best_val={best_val:.6f} @ {best_epoch}"
            )
            print(msg)
            log_f.write(msg + "\n")

            stats_writer.writerow(
                [
                    epoch,
                    args.width,
                    args.depth,
                    neurons,
                    train_loss,
                    train_cls,
                    val_loss,
                    val_cls,
                    val_acc,
                    best_val,
                    best_epoch,
                ]
            )
            stats_f.flush()

            if epochs_no_improve >= args.patience:
                stop_msg = f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)"
                print(stop_msg)
                log_f.write(stop_msg + "\n")
                break
    finally:
        log_f.flush()
        stats_f.flush()

    test_acc = eval_test_accuracy(model, loader, noise_std=args.noise_std)

    report = {
        "graph_path": args.graph_path,
        "in_dim": in_dim,
        "width": args.width,
        "depth": args.depth,
        "neurons_metric": neurons,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "lambda_recon": args.lambda_recon,
        "noise_std": args.noise_std,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log_f.write("\n" + json.dumps(report, indent=2) + "\n")
    log_f.close()
    stats_f.close()


if __name__ == "__main__":
    main()

