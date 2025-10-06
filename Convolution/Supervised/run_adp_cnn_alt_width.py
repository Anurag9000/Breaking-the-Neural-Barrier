
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from adp_cnn_alt_width import TrainConfig, SearchConfig, alternating_adp_search

def make_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int, download: bool) -> Tuple[DataLoader, DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    tf_eval = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_full = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=tf_train, download=download)
    val_full = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=tf_eval, download=False)
    test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=tf_eval, download=download)

    # Deterministic split (10% val)
    g = torch.Generator().manual_seed(42)
    ntrain = int(0.9 * len(train_full))
    idxs = torch.randperm(len(train_full), generator=g)
    train_subset = torch.utils.data.Subset(train_full, idxs[:ntrain])
    val_subset = torch.utils.data.Subset(val_full, idxs[ntrain:])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, persistent_workers=num_workers > 0)
    return train_loader, val_loader, test_loader

@torch.no_grad()
def eval_top1(model, loader, device):
    model.eval().to(device)
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / max(1, total)

def main():
    p = argparse.ArgumentParser("Alternating ADP CNN (Width<->Depth) for CIFAR-10")
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--download", action="store_true")

    # Train config
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--es-patience", type=int, default=100)
    p.add_argument("--max-epochs-inner", type=int, default=1000000)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Search config
    p.add_argument("--delta", type=float, default=0)
    p.add_argument("--patience-width", type=int, default=100)
    p.add_argument("--patience-depth", type=int, default=100)
    p.add_argument("--max-total-epochs", type=int, default=1000000)
    p.add_argument("--pool-idx", type=int, nargs="*", default=[0])

    # Seeds
    p.add_argument("--seed-width", type=int, default=1)
    p.add_argument("--seed-depth", type=int, default=1)

    args = p.parse_args()

    train_loader, val_loader, test_loader = make_cifar10_loaders(
        data_dir=args.data, batch_size=args.batch_size, num_workers=args.num_workers, download=args.download
    )

    tcfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs_inner=args.max_epochs_inner,
        es_patience=args.es_patience,
        grad_clip=args.grad_clip,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    scfg = SearchConfig(
        delta=args.delta,
        patience_width=args.patience_width,
        patience_depth=args.patience_depth,
        max_total_epochs=args.max_total_epochs,
        pooling_indices=tuple(args.pool_idx),
    )

    model, best_val, spent = alternating_adp_search(
        in_ch=3, num_classes=10,
        train_loader=train_loader, val_loader=val_loader,
        tcfg=tcfg, scfg=scfg,
        seed_width=args.seed_width, seed_depth=args.seed_depth,
    )

    device = tcfg.device
    test_acc = eval_top1(model, test_loader, device)

    print("=== Alternating ADP CNN (W<->D) ===")
    print(f"Best Val Loss : {best_val:.4f}")
    print(f"Total Epochs  : {spent}")
    print(f"Final Depth   : {len(model.widths)}")
    print(f"Final Widths  : {model.widths}")
    print(f"Test@1 Acc    : {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
