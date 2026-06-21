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

from ..Models.dae_sparse_mlp_sup_stl import SupDAESparseMLP, sup_dae_total_neurons


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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, num_classes


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def train_one_epoch(
    model: SupDAESparseMLP,
    loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    noise_std: float,
    lambda_recon: float,
    lambda_sparse: float,
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
        xb_rec, logits, z = model(xb_noisy)
        loss_recon = mse(xb_rec, xb)
        loss_cls = ce(logits, yb)
        loss_sparse = z.abs().mean()
        loss = lambda_recon * loss_recon + loss_cls + lambda_sparse * loss_sparse
        loss.backward()
        opt.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_cls += float(loss_cls.item()) * bs
        n += bs
    return total_loss / max(n, 1), total_cls / max(n, 1)


def eval_epoch(
    model: SupDAESparseMLP,
    loader: DataLoader,
    device: torch.device,
    noise_std: float,
    lambda_recon: float,
    lambda_sparse: float,
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
            xb_rec, logits, z = model(xb_noisy)

            loss_recon = mse(xb_rec, xb) / xb.size(0)
            loss_cls = ce(logits, yb) / xb.size(0)
            loss_sparse = z.abs().mean()
            loss = lambda_recon * loss_recon + loss_cls + lambda_sparse * loss_sparse

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total_cls += float(loss_cls.item()) * bs
            n += bs
    return total_loss / max(n, 1), total_cls / max(n, 1)


def eval_accuracy(model: SupDAESparseMLP, loader: DataLoader, device: torch.device, noise_std: float) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, noise_std)
            _, logits, _ = model(xb_noisy)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += xb.size(0)
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Supervised sparse MLP DAE encoder + classifier")
    # Data / IO
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./Runs/dae_sparse_mlp_sup")
    p.add_argument("--seed", type=int, default=1337)
    # Model
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    # Train
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_sparse", type=float, default=1e-3)

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

    model = SupDAESparseMLP(
        num_classes=num_classes,
        in_channels=3,
        img_size=32,
        width=args.width,
        depth=args.depth,
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
        [
            "epoch",
            "width",
            "depth",
            "neurons",
            "train_loss",
            "train_cls",
            "val_loss",
            "val_cls",
            "val_acc",
            "best_val",
            "best_epoch",
        ]
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    neurons = sup_dae_total_neurons(args.width, args.depth, num_classes)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_cls = train_one_epoch(
                model,
                train_loader,
                opt,
                device,
                noise_std=args.noise_std,
                lambda_recon=args.lambda_recon,
                lambda_sparse=args.lambda_sparse,
            )
            val_loss, val_cls = eval_epoch(
                model,
                val_loader,
                device,
                noise_std=args.noise_std,
                lambda_recon=args.lambda_recon,
                lambda_sparse=args.lambda_sparse,
            )
            val_acc = eval_accuracy(model, val_loader, device, noise_std=args.noise_std)

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
                f"Epoch {epoch:03d} | train={train_loss:.4f} (cls={train_cls:.4f}) | "
                f"val={val_loss:.4f} (cls={val_cls:.4f}, acc={val_acc:.4f}) | "
                f"best_val={best_val:.4f} @ {best_epoch}"
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

    # Load best and evaluate on test set
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    test_acc = eval_accuracy(model, test_loader, device, noise_std=args.noise_std)

    report = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "width": args.width,
        "depth": args.depth,
        "neurons_metric": neurons,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "lambda_recon": args.lambda_recon,
        "lambda_sparse": args.lambda_sparse,
        "noise_std": args.noise_std,
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

