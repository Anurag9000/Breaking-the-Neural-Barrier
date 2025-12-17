import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from DAE.Supervised.Models.dae_metric_conv_sup_stl import (
    SupDAEGaussianMetricConv,
    sup_metric_dae_total_neurons,
)


def build_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    val_frac: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    ds_name = dataset.lower()
    if ds_name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ds_cls = datasets.CIFAR100
        num_classes = 100
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_cls = datasets.CIFAR10
        num_classes = 10

    tf_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    tf_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    full_train = ds_cls(root=data_root, train=True, transform=tf_train, download=True)
    test_ds = ds_cls(root=data_root, train=False, transform=tf_test, download=True)

    n_val = int(len(full_train) * val_frac)
    n_train = len(full_train) - n_val
    g = torch.Generator().manual_seed(seed)
    ds_train, ds_val = random_split(full_train, [n_train, n_val], generator=g)

    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    dl_test = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return dl_train, dl_val, dl_test, num_classes


def supervised_contrastive_loss(
    z: torch.Tensor, y: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    if z.ndim != 2:
        z = z.view(z.size(0), -1)
    z = F.normalize(z, dim=1)
    logits = torch.matmul(z, z.t()) / temperature

    labels = y.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.t()).float()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss


def train_one_epoch(
    model: SupDAEGaussianMetricConv,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    lambda_recon: float,
    lambda_contrast: float,
    temperature: float,
    noise_std: float,
) -> float:
    model.train()
    mse = nn.MSELoss()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        if noise_std > 0.0:
            xb_noisy = xb + torch.randn_like(xb) * noise_std
        else:
            xb_noisy = xb

        xb_rec, z = model(xb_noisy)
        loss_recon = mse(xb_rec, xb)
        loss_metric = supervised_contrastive_loss(z, yb, temperature)
        loss = lambda_recon * loss_recon + lambda_contrast * loss_metric
        loss.backward()
        opt.step()

        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def eval_epoch(
    model: SupDAEGaussianMetricConv,
    loader: DataLoader,
    device: torch.device,
    lambda_recon: float,
    lambda_contrast: float,
    temperature: float,
    noise_std: float,
) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if noise_std > 0.0:
                xb_noisy = xb + torch.randn_like(xb) * noise_std
            else:
                xb_noisy = xb
            xb_rec, z = model(xb_noisy)
            loss_recon = mse(xb_rec, xb) / xb.size(0)
            loss_metric = supervised_contrastive_loss(z, yb, temperature)
            loss = lambda_recon * loss_recon + lambda_contrast * loss_metric
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="runs/dae_metric_conv_sup_stl")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--pool-after", type=str, default="2")

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-contrast", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.1)

    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_after = [] if args.pool_after.strip() == "" else [int(x) for x in args.pool_after.split(",")]

    dl_train, dl_val, dl_test, num_classes = build_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    model = SupDAEGaussianMetricConv(
        proj_dim=args.proj_dim,
        in_channels=3,
        width=args.width,
        depth=args.depth,
        pool_after=pool_after,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    log_path = out_dir / "training_log.txt"
    stats_path = out_dir / "training_stats.csv"
    log_f = log_path.open("w", encoding="utf-8")
    stats_f = stats_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(stats_f)
    writer.writerow(
        [
            "epoch",
            "width",
            "depth",
            "proj_dim",
            "neurons",
            "train_loss",
            "val_loss",
            "best_val",
            "best_epoch",
        ]
    )

    best_val = float("inf")
    best_epoch = -1
    es_counter = 0
    ckpt_path = out_dir / "best.pt"
    neurons = sup_metric_dae_total_neurons(args.width, args.depth, args.proj_dim)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                dl_train,
                opt,
                device,
                args.lambda_recon,
                args.lambda_contrast,
                args.temperature,
                args.noise_std,
            )
            val_loss = eval_epoch(
                model,
                dl_val,
                device,
                args.lambda_recon,
                args.lambda_contrast,
                args.temperature,
                args.noise_std,
            )

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
                    ckpt_path,
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
                [
                    epoch,
                    args.width,
                    args.depth,
                    args.proj_dim,
                    neurons,
                    train_loss,
                    val_loss,
                    best_val,
                    best_epoch,
                ]
            )
            stats_f.flush()

            if es_counter >= args.patience:
                stop_msg = (
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)"
                )
                print(stop_msg)
                log_f.write(stop_msg + "\n")
                break
    finally:
        log_f.flush()
        stats_f.flush()

    # Evaluate reconstruction quality on test set (same loss)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    test_loss = eval_epoch(
        model,
        dl_test,
        device,
        args.lambda_recon,
        args.lambda_contrast,
        args.temperature,
        args.noise_std,
    )

    report = {
        "dataset": args.dataset,
        "width": args.width,
        "depth": args.depth,
        "proj_dim": args.proj_dim,
        "neurons": neurons,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

