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

from ..Models.dae_stacked_mlp_stl import DAEStackedMLP, dae_total_neurons


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

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    return train_loader, val_loader


def add_gaussian_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return x
    return x + torch.randn_like(x) * noise_std


def layerwise_pretrain(
    model: DAEStackedMLP,
    loader: DataLoader,
    device: torch.device,
    noise_std: float,
    epochs: int,
    lr: float,
) -> None:
    if epochs <= 0:
        return

    enc_linears = [m for m in model.encoder if isinstance(m, nn.Linear)]
    dec_linears = [m for m in model.decoder if isinstance(m, nn.Linear)][::-1]

    mse = nn.MSELoss()
    for layer_idx, (enc_lin, dec_lin) in enumerate(zip(enc_linears, dec_linears)):
        auto = nn.Sequential(
            nn.Linear(enc_lin.in_features, enc_lin.out_features),
            nn.ReLU(inplace=True),
            nn.Linear(enc_lin.out_features, enc_lin.in_features),
        ).to(device)

        with torch.no_grad():
            auto[0].weight.copy_(enc_lin.weight)
            if enc_lin.bias is not None and auto[0].bias is not None:
                auto[0].bias.copy_(enc_lin.bias)
            auto[2].weight.copy_(dec_lin.weight)
            if dec_lin.bias is not None and auto[2].bias is not None:
                auto[2].bias.copy_(dec_lin.bias)

        opt = optim.Adam(auto.parameters(), lr=lr)

        for _ in range(epochs):
            for xb, _ in loader:
                xb = xb.to(device, non_blocking=True)
                noise = torch.randn_like(xb) * noise_std if noise_std > 0 else 0.0
                noisy = xb + noise
                clean_flat = xb.view(xb.size(0), model.input_dim)
                noisy_flat = noisy.view(noisy.size(0), model.input_dim)

                with torch.no_grad():
                    clean_in = model.forward_activations(clean_flat, layer_idx)
                    noisy_in = model.forward_activations(noisy_flat, layer_idx)

                opt.zero_grad(set_to_none=True)
                recon = auto(noisy_in)
                loss = mse(recon, clean_in)
                loss.backward()
                opt.step()

        with torch.no_grad():
            enc_lin.weight.copy_(auto[0].weight)
            if enc_lin.bias is not None and auto[0].bias is not None:
                enc_lin.bias.copy_(auto[0].bias)
            dec_lin.weight.copy_(auto[2].weight)
            if dec_lin.bias is not None and auto[2].bias is not None:
                dec_lin.bias.copy_(auto[2].bias)


def train_one_epoch(
    model: DAEStackedMLP,
    loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    noise_std: float,
) -> float:
    model.train()
    mse = nn.MSELoss()
    total, n = 0.0, 0
    for xb, _ in loader:
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
    model: DAEStackedMLP,
    loader: DataLoader,
    device: torch.device,
    noise_std: float,
) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, noise_std)
            xb_rec, _ = model(xb_noisy)
            total += float(mse(xb_rec, xb).item())
            n += xb.size(0)
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Stacked (layer-wise pretrain) MLP DAE STL on CIFAR")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./Runs/dae_stacked_mlp_stl")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--pretrain_epochs", type=int, default=5)
    p.add_argument("--pretrain_lr", type=float, default=1e-3)

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

    model = DAEStackedMLP(in_channels=3, img_size=32, width=args.width, depth=args.depth).to(device)

    if args.pretrain_epochs > 0:
        layerwise_pretrain(
            model,
            train_loader,
            device,
            noise_std=args.noise_std,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
        )

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
    neurons = dae_total_neurons(args.width, args.depth)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, opt, device, noise_std=args.noise_std)
            val_loss = eval_epoch(model, val_loader, device, noise_std=args.noise_std)

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
        "dataset": args.dataset,
        "width": args.width,
        "depth": args.depth,
        "neurons_metric": neurons,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "noise_std": args.noise_std,
        "pretrain_epochs": args.pretrain_epochs,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log_f.write("\n" + json.dumps(report, indent=2) + "\n")
    log_f.close()
    stats_f.close()


if __name__ == "__main__":
    try:
        import os as _os, sys as _sys
        if _os.name == "posix" and _sys.platform.startswith("linux"):
            import ctypes as _ctypes
            _ctypes.CDLL("libc.so.6", use_errno=True).mlockall(3)
        elif _os.name == "nt":
            import ctypes as _ctypes
            _ctypes.windll.kernel32.SetProcessWorkingSetSize(_ctypes.windll.kernel32.GetCurrentProcess(), -1, -1)
    except Exception:
        pass
    main()

