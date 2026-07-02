import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision as tv
import torchvision.transforms as T

from ..Models.dae_unet_conv_sup_stl import SupDAEUNetConv, sup_dae_total_neurons


def build_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_frac: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    g = torch.Generator().manual_seed(seed)

    if dataset.lower() == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_class = tv.datasets.CIFAR10
        num_classes = 10
    elif dataset.lower() == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ds_class = tv.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    tf_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    tf_eval = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    train_full = ds_class(root=data_dir, train=True, transform=tf_train, download=True)
    test_ds = ds_class(root=data_dir, train=False, transform=tf_eval, download=True)

    val_size = int(len(train_full) * val_frac)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, test_loader, num_classes


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def train_one_epoch(
    model: SupDAEUNetConv,
    loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    noise_std: float,
    lambda_recon: float,
) -> Tuple[float, float]:
    model.train()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    total_loss, total_cls, n = 0.0, 0.0, 0
    for xb, yb in loader:
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
    model: SupDAEUNetConv,
    loader: DataLoader,
    device: torch.device,
    noise_std: float,
    lambda_recon: float,
) -> Tuple[float, float]:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_cls, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, noise_std)
            xb_rec, logits = model(xb_noisy)
            loss_recon = mse(xb_rec, xb)
            loss_cls = ce(logits, yb)
            loss = lambda_recon * loss_recon + loss_cls
            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total_cls += float(loss_cls.item()) * bs
            n += bs
    return total_loss / max(n, 1), total_cls / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Supervised U-Net Conv DAE encoder + CIFAR classifier")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="results_dae_unet_conv_sup")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--lambda-recon", type=float, default=1.0)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10000000000000000000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, num_classes = build_dataloaders(
        args.dataset,
        args.data_root,
        args.batch_size,
        args.num_workers,
        val_frac=0.1,
        seed=args.seed,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = SupDAEUNetConv(num_classes=num_classes, in_channels=3, width=args.width, depth=args.depth).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_f = (results_dir / "training_log.txt").open("w", encoding="utf-8")
    stats_f = (results_dir / "training_stats.csv").open("w", newline="", encoding="utf-8")
    stats_writer = csv.writer(stats_f)
    stats_writer.writerow(
        ["epoch", "width", "depth", "neurons", "train_loss", "cls_loss", "val_loss", "val_cls_loss", "best_val", "best_epoch"]
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_bad = 0
    ckpt_path = results_dir / "best.pt"

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
                epochs_bad = 0
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "args": vars(args)},
                    ckpt_path,
                )
            else:
                epochs_bad += 1

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

            if epochs_bad >= args.patience:
                stop = f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)"
                print(stop)
                log_f.write(stop + "\n")
                break
    finally:
        log_f.flush()
        stats_f.flush()

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    test_loss, test_cls = eval_epoch(
        model, test_loader, device, args.noise_std, args.lambda_recon
    )

    report = {
        "dataset": args.dataset,
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
