import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..Models.dae_rnn_seq_stl import DAERNNSeq, rnn_total_neurons


class SequenceDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        if data.dim() == 2:
            data = data.unsqueeze(1)
        assert data.dim() == 3, "Expected data of shape (N, C, L) or (N, L)"
        self.data = data

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def load_sequences(path: Path) -> torch.Tensor:
    obj = torch.load(path)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], torch.Tensor):
        return obj["data"]
    raise RuntimeError(f"Unsupported sequence format in {path}")


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def train_one_epoch(
    model: DAERNNSeq,
    dl: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float,
) -> float:
    model.train()
    mse = nn.MSELoss()
    total, n = 0.0, 0
    for xb in dl:
        xb = xb.to(device, non_blocking=True)
        xb_noisy = add_gaussian_noise(xb, noise_std)

        opt.zero_grad(set_to_none=True)
        xb_rec, _ = model(xb_noisy)
        loss = mse(xb_rec, xb)
        loss.backward()
        opt.step()

        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def eval_epoch(
    model: DAERNNSeq,
    dl: DataLoader,
    device: torch.device,
    noise_std: float,
) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, noise_std)
            xb_rec, _ = model(xb_noisy)
            total += float(mse(xb_rec, xb).item())
            n += xb.size(0)
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True, help="Path to .pt tensor or dict with key 'data'")
    p.add_argument("--results-dir", type=str, default="results_dae_rnn_seq")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--rnn-type", type=str, default="gru", choices=["gru", "lstm"])
    p.add_argument("--noise-std", type=float, default=0.1)
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
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.data_path)
    raw = load_sequences(data_path)
    dataset = SequenceDataset(raw)

    n_total = len(dataset)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    in_channels = dataset[0].shape[0]
    model = DAERNNSeq(in_channels=in_channels, width=args.width, depth=args.depth, rnn_type=args.rnn_type).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / "training_log.txt"
    csv_path = results_dir / "training_stats.csv"

    log_f = log_path.open("w", encoding="utf-8")
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(
        ["epoch", "width", "depth", "neurons", "train_loss", "val_loss", "best_val", "best_epoch"]
    )

    neurons_metric = rnn_total_neurons(args.width, args.depth)
    best_val = float("inf")
    best_epoch = -1
    es_counter = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, dl_train, opt, device, args.noise_std)
            val_loss = eval_epoch(model, dl_val, device, args.noise_std)

            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                best_epoch = epoch
                es_counter = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "args": vars(args),
                    },
                    results_dir / "best.pt",
                )
            else:
                es_counter += 1

            msg = (
                f"Epoch {epoch:03d} | train={train_loss:.6f} | "
                f"val={val_loss:.6f} | best_val={best_val:.6f} @ {best_epoch}"
            )
            print(msg)
            log_f.write(msg + "\n")

            writer.writerow(
                [epoch, args.width, args.depth, neurons_metric, train_loss, val_loss, best_val, best_epoch]
            )
            csv_f.flush()

            if es_counter >= args.patience:
                stop_msg = f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)"
                print(stop_msg)
                log_f.write(stop_msg + "\n")
                break
    finally:
        log_f.flush()
        csv_f.flush()

    report = {
        "data_path": str(data_path),
        "in_channels": in_channels,
        "width": args.width,
        "depth": args.depth,
        "rnn_type": args.rnn_type,
        "noise_std": args.noise_std,
        "neurons_metric": neurons_metric,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
    }
    with (results_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

