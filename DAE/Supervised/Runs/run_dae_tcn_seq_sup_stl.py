import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..Models.dae_tcn_seq_sup_stl import SupDAETCNSeq, sup_dae_total_neurons


class LabeledSequenceDataset(Dataset):
    """
    Supervised wrapper around a tensor of sequences and labels.

    Expects:
      - data: (N, C, L) or (N, L)
      - labels: (N,)
    """

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        if data.dim() == 2:
            data = data.unsqueeze(1)
        assert data.dim() == 3, "Expected data of shape (N, C, L) or (N, L)"
        assert labels.dim() == 1 and labels.size(0) == data.size(0)
        self.data = data
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


def load_supervised_sequences(path: Path) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Loads (data, labels, num_classes) from a .pt file.
    Supported formats:
      - dict with keys: 'data' (tensor), 'labels' (tensor), optional 'num_classes'
    """
    obj = torch.load(path)
    if not isinstance(obj, dict) or "data" not in obj or "labels" not in obj:
        raise RuntimeError("Expected a dict with keys 'data' and 'labels' in supervised sequence file")
    data = obj["data"]
    labels = obj["labels"]
    num_classes = int(obj.get("num_classes", int(labels.max().item()) + 1))
    return data, labels, num_classes


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def train_one_epoch(
    model: SupDAETCNSeq,
    dl: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float,
    lambda_recon: float,
) -> Tuple[float, float]:
    model.train()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    total_loss, total_cls, n = 0.0, 0.0, 0
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xb_noisy = add_gaussian_noise(xb, noise_std)

        opt.zero_grad(set_to_none=True)
        xb_rec, logits = model(xb_noisy)
        loss_recon = mse(xb_rec, xb)
        loss_cls = ce(logits, yb)
        loss = lambda_recon * loss_recon + loss_cls
        loss.backward()
        opt.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_cls += float(loss_cls.item()) * bs
        n += bs
    return total_loss / max(n, 1), total_cls / max(n, 1)


def eval_epoch(
    model: SupDAETCNSeq,
    dl: DataLoader,
    device: torch.device,
    noise_std: float,
    lambda_recon: float,
) -> Tuple[float, float]:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_cls, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, noise_std)
            xb_rec, logits = model(xb_noisy)
            loss_recon = mse(xb_rec, xb)
            loss_cls = ce(logits, yb)
            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total_cls += float(loss_cls.item()) * bs
            n += bs
    return total_loss / max(n, 1), total_cls / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Supervised temporal TCN DAE encoder + classifier")
    p.add_argument("--data-path", type=str, required=True, help="Path to .pt dict with 'data' and 'labels'")
    p.add_argument("--results-dir", type=str, default="results_dae_tcn_seq_sup")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--lambda-recon", type=float, default=1.0)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10000000000000000000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.data_path)
    data, labels, num_classes = load_supervised_sequences(data_path)

    full_ds = LabeledSequenceDataset(data, labels)
    val_size = int(len(full_ds) * args.val_frac)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = SupDAETCNSeq(num_classes=num_classes, in_channels=full_ds.data.size(1), width=args.width, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_f = (results_dir / "training_log.txt").open("w", encoding="utf-8")
    stats_f = (results_dir / "training_stats.csv").open("w", newline="", encoding="utf-8")
    stats_writer = csv.writer(stats_f)
    stats_writer.writerow(
        ["epoch", "width", "depth", "neurons", "train_loss", "train_cls", "val_loss", "val_cls", "best_val", "best_epoch"]
    )

    best_val = float("inf")
    best_epoch = -1
    bad = 0
    ckpt = results_dir / "best.pt"

    neurons = sup_dae_total_neurons(args.width, args.depth, num_classes)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_cls = train_one_epoch(
                model, train_loader, opt, device, args.noise_std, args.lambda_recon
            )
            val_loss, val_cls = eval_epoch(
                model, val_loader, device, args.noise_std, args.lambda_recon
            )

            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                best_epoch = epoch
                bad = 0
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "args": vars(args)},
                    ckpt,
                )
            else:
                bad += 1

            msg = (
                f"Epoch {epoch:03d} | train={train_loss:.4f} | train_cls={train_cls:.4f} | "
                f"val={val_loss:.4f} | val_cls={val_cls:.4f} | best_val={best_val:.4f} @ {best_epoch}"
            )
            print(msg)
            log_f.write(msg + "\n")
            stats_writer.writerow(
                [epoch, args.width, args.depth, neurons, train_loss, train_cls, val_loss, val_cls, best_val, best_epoch]
            )
            stats_f.flush()

            if bad >= args.patience:
                stop = f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)"
                print(stop)
                log_f.write(stop + "\n")
                break
    finally:
        log_f.flush()
        stats_f.flush()

    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
    test_loss, test_cls = eval_epoch(
        model, test_loader, device, args.noise_std, args.lambda_recon
    )

    report = {
        "data_path": str(data_path),
        "width": args.width,
        "depth": args.depth,
        "neurons_metric": neurons,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_cls_loss": test_cls,
    }
    with (results_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

