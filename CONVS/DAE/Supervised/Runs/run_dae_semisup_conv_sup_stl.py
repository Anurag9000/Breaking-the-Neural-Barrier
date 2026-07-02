import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision as tv
import torchvision.transforms as T
from itertools import zip_longest

from ..Models.dae_semisup_conv_sup_stl import SupDAESemiConv, sup_dae_semisup_total_neurons


def build_dataloaders_semisup(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_frac: float,
    label_frac: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int]:
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

    full = ds_class(root=data_dir, train=True, transform=tf_train, download=True)
    test_ds = ds_class(root=data_dir, train=False, transform=tf_eval, download=True)

    indices = torch.randperm(len(full), generator=g).tolist()
    val_size = int(len(full) * val_frac)
    val_idx = indices[:val_size]
    train_idx_all = indices[val_size:]

    n_label = int(len(train_idx_all) * label_frac)
    label_idx = train_idx_all[:n_label]
    unlabel_idx = train_idx_all[n_label:] if n_label < len(train_idx_all) else []

    labeled_ds = Subset(full, label_idx)
    unlabeled_ds = Subset(full, unlabel_idx) if unlabel_idx else Subset(full, [])
    val_ds = Subset(full, val_idx)

    labeled_loader = DataLoader(
        labeled_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    return labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def train_one_epoch(
    model: SupDAESemiConv,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    noise_std: float,
    lambda_recon: float,
) -> Tuple[float, float]:
    model.train()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    total_loss, total_cls, n = 0.0, 0.0, 0

    for lab_batch, unlab_batch in zip_longest(labeled_loader, unlabeled_loader):
        loss = 0.0
        loss_cls = 0.0
        bs_total = 0

        if lab_batch is not None:
            xb_lab, yb_lab = lab_batch
            xb_lab = xb_lab.to(device, non_blocking=True)
            yb_lab = yb_lab.to(device, non_blocking=True)
            xb_lab_noisy = add_gaussian_noise(xb_lab, noise_std)
            x_rec_lab, logits_lab = model(xb_lab_noisy)
            loss_recon_lab = mse(x_rec_lab, xb_lab)
            loss_cls_lab = ce(logits_lab, yb_lab)
            loss = loss + lambda_recon * loss_recon_lab + loss_cls_lab
            loss_cls = loss_cls + loss_cls_lab
            bs_total += xb_lab.size(0)

        if unlab_batch is not None:
            xb_unlab, _ = unlab_batch
            xb_unlab = xb_unlab.to(device, non_blocking=True)
            xb_unlab_noisy = add_gaussian_noise(xb_unlab, noise_std)
            x_rec_unlab, _ = model(xb_unlab_noisy)
            loss_recon_unlab = mse(x_rec_unlab, xb_unlab)
            loss = loss + lambda_recon * loss_recon_unlab
            bs_total += xb_unlab.size(0)

        if bs_total == 0:
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * bs_total
        total_cls += float(loss_cls.item()) * (lab_batch[0].size(0) if lab_batch is not None else 0)  # type: ignore[index]
        n += bs_total

    return total_loss / max(n, 1), total_cls / max(n, 1)


def eval_epoch(
    model: SupDAESemiConv,
    val_loader: DataLoader,
    device: torch.device,
    noise_std: float,
    lambda_recon: float,
) -> Tuple[float, float]:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_cls, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, noise_std)
            x_rec, logits = model(xb_noisy)

            loss_recon = mse(x_rec, xb) / xb.size(0)
            loss_cls = ce(logits, yb) / xb.size(0)
            loss = lambda_recon * loss_recon + loss_cls

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total_cls += float(loss_cls.item()) * bs
            n += bs
    return total_loss / max(n, 1), total_cls / max(n, 1)


def eval_accuracy(model: SupDAESemiConv, loader: DataLoader, device: torch.device, noise_std: float) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, noise_std)
            _, logits = model(xb_noisy)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += xb.size(0)
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Semi-supervised Conv DAE encoder + classifier")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--label_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--lambda_recon", type=float, default=1.0)

    p.add_argument("--out_dir", type=str, default="runs/dae_semisup_conv_sup_cifar")

    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes = build_dataloaders_semisup(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        label_frac=args.label_frac,
        seed=args.seed,
    )

    model = SupDAESemiConv(
        num_classes=num_classes,
        in_channels=3,
        width=args.width,
        depth=args.depth,
        pool_after=[2],
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
    neurons = sup_dae_semisup_total_neurons(args.width, args.depth, num_classes)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_cls = train_one_epoch(
                model,
                labeled_loader,
                unlabeled_loader,
                opt,
                device,
                noise_std=args.noise_std,
                lambda_recon=args.lambda_recon,
            )
            val_loss, val_cls = eval_epoch(
                model,
                val_loader,
                device,
                noise_std=args.noise_std,
                lambda_recon=args.lambda_recon,
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

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    test_acc = eval_accuracy(model, test_loader, device, noise_std=args.noise_std)

    report = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "width": args.width,
        "depth": args.depth,
        "label_frac": args.label_frac,
        "neurons_metric": neurons,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "lambda_recon": args.lambda_recon,
        "noise_std": args.noise_std,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log_f.write("\n" + json.dumps(report, indent=2) + "\n")
    log_f.close()
    stats_f.close()


if __name__ == "__main__":
    main()

