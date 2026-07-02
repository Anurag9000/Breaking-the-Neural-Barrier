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

from ..Models.dae_vae_conv_stl import DAEVAEConv, dae_vae_total_neurons


def build_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_frac: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)

    if dataset.lower() == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_class = tv.datasets.CIFAR10
    elif dataset.lower() == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ds_class = tv.datasets.CIFAR100
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

    full_train = ds_class(root=data_dir, train=True, transform=tf_train, download=True)
    full_eval = ds_class(root=data_dir, train=True, transform=tf_eval, download=True)

    val_size = int(len(full_train) * val_frac)
    train_size = len(full_train) - val_size

    train_ds, _ = random_split(full_train, [train_size, val_size], generator=g)
    _, val_ds = random_split(full_eval, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # standard normal prior
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def train_one_epoch(
    model: DAEVAEConv,
    loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    noise_std: float,
    beta: float,
) -> float:
    model.train()
    mse = nn.MSELoss()
    total, n = 0.0, 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        xb_noisy = add_gaussian_noise(xb, noise_std)

        opt.zero_grad(set_to_none=True)
        xb_rec, mu, logvar = model(xb_noisy)
        recon = mse(xb_rec, xb)
        kld = kl_divergence(mu, logvar)
        loss = recon + beta * kld
        loss.backward()
        opt.step()

        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def eval_epoch(
    model: DAEVAEConv,
    loader: DataLoader,
    device: torch.device,
    noise_std: float,
    beta: float,
) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, noise_std)
            xb_rec, mu, logvar = model(xb_noisy)
            recon = mse(xb_rec, xb) / xb.size(0)
            kld = kl_divergence(mu, logvar)
            loss = recon + beta * kld
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Variational Conv DAE (denoising VAE) STL on CIFAR")
    # Data / IO
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./Runs/dae_vae_conv_stl")
    p.add_argument("--seed", type=int, default=1337)
    # Model
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--latent_dim", type=int, default=128)
    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=1.0)

    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    model = DAEVAEConv(in_channels=3, width=args.width, depth=args.depth, latent_dim=args.latent_dim).to(device)
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
        ["epoch", "width", "depth", "latent_dim", "neurons", "train_loss", "val_loss", "best_val", "best_epoch"]
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    neurons = dae_vae_total_neurons(args.width, args.depth, args.latent_dim)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, opt, device, noise_std=args.noise_std, beta=args.beta
            )
            val_loss = eval_epoch(
                model, val_loader, device, noise_std=args.noise_std, beta=args.beta
            )

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

            stats_writer.writerow(
                [epoch, args.width, args.depth, args.latent_dim, neurons, train_loss, val_loss, best_val, best_epoch]
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

    report = {
        "dataset": args.dataset,
        "width": args.width,
        "depth": args.depth,
        "latent_dim": args.latent_dim,
        "neurons_metric": neurons,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "noise_std": args.noise_std,
        "beta": args.beta,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log_f.write("\n" + json.dumps(report, indent=2) + "\n")
    log_f.close()
    stats_f.close()


if __name__ == "__main__":
    main()

