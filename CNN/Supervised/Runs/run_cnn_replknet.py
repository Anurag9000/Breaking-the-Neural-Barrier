"""
Runner for RepLKNet (CIFAR) — single-model supervised.
Supports presets {tiny, small, base}, optional kernel override, layer scale, and drop-path.
Standard early-stopping loop, deterministic split, AdamW, CE ± label smoothing, optional cosine schedule,
grad-clip, restore best, final test.
"""
from __future__ import annotations
import argparse
import os
import random
from time import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as T

from CNN_RepLKNet import make_replknet_cifar

# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total += images.size(0)
    return total_loss / total, total_correct / total

# ----------------------------
# Main train loop with early stopping
# ----------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Transforms
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)

    train_tfms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(cifar_mean, cifar_std),
    ])
    eval_tfms = T.Compose([
        T.ToTensor(),
        T.Normalize(cifar_mean, cifar_std),
    ])

    # Dataset selection
    if args.dataset.lower() == "cifar100":
        full_train = torchvision.datasets.CIFAR100(root=args.data_root, train=True, transform=train_tfms, download=True)
        full_train_eval = torchvision.datasets.CIFAR100(root=args.data_root, train=True, transform=eval_tfms, download=True)
        test_set = torchvision.datasets.CIFAR100(root=args.data_root, train=False, transform=eval_tfms, download=True)
        num_classes = 100
    else:
        full_train = torchvision.datasets.CIFAR10(root=args.data_root, train=True, transform=train_tfms, download=True)
        full_train_eval = torchvision.datasets.CIFAR10(root=args.data_root, train=True, transform=eval_tfms, download=True)
        test_set = torchvision.datasets.CIFAR10(root=args.data_root, train=False, transform=eval_tfms, download=True)
        num_classes = 10

    # Deterministic split train/val from the TRAIN set
    n_total = len(full_train)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val
    train_set, _ = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    _, val_set = random_split(full_train_eval, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model/opt/loss
    model = make_replknet_cifar(num_classes=num_classes, in_channels=3, preset=args.preset,
                                layer_scale_init=args.layer_scale, drop_path_rate=args.drop_path,
                                k_override=args.kernel if args.kernel > 0 else None).to(device)

    # Label smoothing optional
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing) if args.label_smoothing > 0 else nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.cosine:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)

    # Early stopping state
    best_val = float("inf")
    best_state = None
    best_epoch = -1

    start_time = time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item() * images.size(0)
            seen += images.size(0)
        train_loss = running / max(1, seen)

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        if scheduler is not None:
            scheduler.step()

        improved = val_loss < best_val - args.delta
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch

        print(f"Epoch {epoch:03d}/{args.max_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}% | best_val={best_val:.4f} @ {best_epoch}")

        if (epoch - best_epoch) >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} (val_loss={best_val:.4f}).")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    elapsed = time() - start_time

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        "model": f"RepLKNet({args.preset}, k={args.kernel if args.kernel>0 else 'preset'}, layer_scale={args.layer_scale}, drop_path={args.drop_path})",
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "args": vars(args),
    }, args.save_path)

    print("-"*72)
    print(f"Done in {elapsed/60:.1f} min | best_val_loss={best_val:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}%")
    print(f"Saved checkpoint to: {args.save_path}")


def build_argparser():
    p = argparse.ArgumentParser(description="Train RepLKNet (CIFAR) with early stopping (single-model).")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--save_path", type=str, default="./checkpoints/replknet_tiny_cifar10.pth")

    p.add_argument("--preset", type=str, default="tiny", choices=["tiny","small","base"])
    p.add_argument("--kernel", type=int, default=0, help="Override large kernel size (0 → preset)")
    p.add_argument("--layer_scale", type=float, default=1e-6)
    p.add_argument("--drop_path", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--test_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--cosine", action="store_true", help="Use CosineAnnealingLR")

    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--delta", type=float, default=1e-4, help="Min val-loss improvement to count as better")
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--label_smoothing", type=float, default=0.0)

    p.add_argument("--val_frac", type=float, default=0.1, help="Fraction of TRAIN set held out for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
