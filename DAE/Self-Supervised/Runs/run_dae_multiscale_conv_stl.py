import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ..Models.dae_multiscale_conv_stl import DAEMultiScaleConv, dae_total_neurons


def make_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    ds_name = dataset.lower()
    if ds_name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ds_cls = datasets.CIFAR100
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_cls = datasets.CIFAR10

    tf_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    tf_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    full_train = ds_cls(root=data_root, train=True, transform=tf_train, download=True)
    full_eval = ds_cls(root=data_root, train=True, transform=tf_eval, download=True)

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_ds, _ = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(1337))
    _, val_ds = random_split(full_eval, [n_train, n_val], generator=torch.Generator().manual_seed(1337))

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def train_one_epoch(
    model: DAEMultiScaleConv,
    dl: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float,
) -> float:
    model.train()
    mse = nn.MSELoss()
    total, n = 0.0, 0
    for xb, _ in dl:
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
    model: DAEMultiScaleConv,
    dl: DataLoader,
    device: torch.device,
    noise_std: float,
) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, noise_std)
            xb_rec, _ = model(xb_noisy)
            total += float(mse(xb_rec, xb).item())
            n += xb.size(0)
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="results_dae_multiscale_conv_cifar")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10000000000000000000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)

    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dl_train, dl_val = make_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    model = DAEMultiScaleConv(in_channels=3, width=args.width, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = results_dir / "training_log.txt"
    csv_path = results_dir / "training_stats.csv"

    log_f = log_path.open("w", encoding="utf-8")
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(
        ["epoch", "width", "depth", "neurons", "train_loss", "val_loss", "best_val", "best_epoch"]
    )

    neurons_metric = dae_total_neurons(args.width, args.depth)
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
        "dataset": args.dataset,
        "width": args.width,
        "depth": args.depth,
        "neurons_metric": neurons_metric,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
    }
    with (results_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

