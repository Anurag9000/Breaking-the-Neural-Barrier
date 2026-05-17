"""
Runner for Curriculum DAE classifier.
Implements faithful Curriculum Learning (Easy-to-Hard noise schedule).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Reuse components from Gaussian DAE
from ..Models.dae_gaussian_conv_sup_stl import SupDAEGaussianConv, sup_dae_total_neurons
from .run_dae_gaussian_conv_sup_stl import build_dataloaders, add_gaussian_noise, eval_epoch, eval_accuracy

def get_current_noise(epoch: int, max_epochs: int, target_std: float, frac: float) -> float:
    warmup = int(max_epochs * frac)
    if warmup < 1: return target_std
    if epoch >= warmup: return target_std
    return target_std * (epoch / warmup)

def train_one_epoch_curriculum(
    model: SupDAEGaussianConv,
    loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    current_sigma: float,
    lambda_recon: float,
) -> Tuple[float, float]:
    model.train()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    total_loss, total_recon, total_cls, n = 0.0, 0.0, 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        
        # Dynamic Noise
        xb_noisy = add_gaussian_noise(xb, current_sigma)

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

def main() -> None:
    p = argparse.ArgumentParser(description="Curriculum DAE (Easy-to-Hard) on CIFAR")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./Runs/dae_curriculum_sup_stl")
    p.add_argument("--seed", type=int, default=1337)
    # Model
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--pool_after", type=str, default="2")
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    # Curriculum
    p.add_argument("--target_noise_std", type=float, default=0.2)
    p.add_argument("--curriculum_frac", type=float, default=0.5, help="Fraction of epochs for linearly increasing noise")

    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, num_classes = build_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    pool_after = [] if args.pool_after.strip() == "" else [int(x) for x in args.pool_after.split(",")]
    model = SupDAEGaussianConv(
        num_classes=num_classes,
        in_channels=3,
        width=args.width,
        depth=args.depth,
        pool_after=pool_after,
    ).to(device)
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
        ["epoch", "sigma", "train_loss", "val_loss", "val_acc", "best_val"]
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    try:
        for epoch in range(1, args.epochs + 1):
            curr_sigma = get_current_noise(epoch, args.epochs, args.target_noise_std, args.curriculum_frac)
            
            train_loss, train_cls = train_one_epoch_curriculum(
                model,
                train_loader,
                opt,
                device,
                current_sigma=curr_sigma,
                lambda_recon=args.lambda_recon,
            )
            # Evaluate at TARGET noise for consistency
            val_loss, val_cls = eval_epoch(
                model,
                val_loader,
                device,
                noise_std=args.target_noise_std,
                lambda_recon=args.lambda_recon,
            )
            val_acc = eval_accuracy(model, val_loader, device, noise_std=args.target_noise_std)

            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 1

            msg = (
                f"Epoch {epoch:03d} | Sigma={curr_sigma:.3f} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} (acc={val_acc:.4f}) | "
                f"best={best_val:.4f}"
            )
            print(msg)
            log_f.write(msg + "\n")
            stats_writer.writerow([epoch, curr_sigma, train_loss, val_loss, val_acc, best_val])
            stats_f.flush()

            if epochs_no_improve >= args.patience:
                print("Early stopping")
                break
    finally:
        log_f.close()
        stats_f.close()

if __name__ == "__main__":
    main()

