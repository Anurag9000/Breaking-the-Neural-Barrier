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

from ..Models.dae_blockmask_mlp_stl import DAEBlockMaskMLP, dae_total_neurons


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


def add_block_mask_noise(x: torch.Tensor, block_frac: float) -> torch.Tensor:
    """
    Apply a single contiguous spatial block mask per image.
    `block_frac` is the approximate fraction of the image area to mask.
    """
    if block_frac <= 0.0:
        return x

    b, c, h, w = x.shape
    area = h * w
    block_area = max(1, int(area * block_frac))
    # make block roughly square
    side = max(1, int(block_area ** 0.5))
    bh = min(h, side)
    bw = min(w, side)

    x_noisy = x.clone()
    for i in range(b):
        top = torch.randint(0, max(1, h - bh + 1), (1,), device=x.device).item()
        left = torch.randint(0, max(1, w - bw + 1), (1,), device=x.device).item()
        x_noisy[i, :, top : top + bh, left : left + bw] = 0.0
    return x_noisy


def train_one_epoch(
    model: DAEBlockMaskMLP,
    loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    block_frac: float,
) -> float:
    model.train()
    mse = nn.MSELoss()
    total, n = 0.0, 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        xb_noisy = add_block_mask_noise(xb, block_frac)

        opt.zero_grad(set_to_none=True)
        xb_rec, _ = model(xb_noisy)
        loss = mse(xb_rec, xb)
        loss.backward()
        opt.step()

        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def eval_epoch(
    model: DAEBlockMaskMLP,
    loader: DataLoader,
    device: torch.device,
    block_frac: float,
) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            xb_noisy = add_block_mask_noise(xb, block_frac)
            xb_rec, _ = model(xb_noisy)
            total += float(mse(xb_rec, xb).item())
            n += xb.size(0)
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Block-mask MLP DAE STL on CIFAR")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./Runs/dae_blockmask_mlp_stl")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--block_frac", type=float, default=0.15)

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

    model = DAEBlockMaskMLP(in_channels=3, img_size=32, width=args.width, depth=args.depth).to(device)
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
            train_loss = train_one_epoch(model, train_loader, opt, device, block_frac=args.block_frac)
            val_loss = eval_epoch(model, val_loader, device, block_frac=args.block_frac)

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
        "block_frac": args.block_frac,
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
