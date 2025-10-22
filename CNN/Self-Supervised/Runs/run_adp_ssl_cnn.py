import argparse, os, random, torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from adp_ssl_cnn_model import AdaptiveCNN, VARIANT_FUNCS

# Simple dataset factory
def get_dataset(name, train=True):
    name = name.lower()
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])
    if name == "cifar10":
        ds = datasets.CIFAR10(root="./data", train=train, download=True, transform=tf_train if train else tf_eval)
    elif name == "cifar100":
        ds = datasets.CIFAR100(root="./data", train=train, download=True, transform=tf_train if train else tf_eval)
    elif name == "stl10":
        split = "train" if train else "test"
        ds = datasets.STL10(root="./data", split=split, download=True, transform=tf_train if train else tf_eval)
    else:
        raise ValueError("Unknown dataset")
    return ds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","cifar100","stl10"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3, help="inner epochs per proposal")
    p.add_argument("--objective", type=str, default="rotation",
                   choices=["rotation","jigsaw","context","masked_ae","colorization","whitening","rot_jigsaw",
                            "temporal_order","optical_flow","exemplar","predictive_coding"])
    p.add_argument("--variant", type=str, default="wd",
                   choices=["wd","dw","alt_d","alt_w","depth_only","width_only"])
    p.add_argument("--widths", type=str, default="32,32,64")
    p.add_argument("--pool_idx", type=str, default="1", help="0-based comma-separated indices for MaxPool")
    p.add_argument("--ex_k", type=int, default=16)
    p.add_argument("--patience_depth", type=int, default=2)
    p.add_argument("--patience_width", type=int, default=2)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--max_width", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = get_dataset(args.dataset, train=True)
    # held-out val split
    val_frac = 0.1
    n_val = int(len(train_ds)*val_frac)
    n_train = len(train_ds)-n_val
    train_ds, val_ds = random_split(train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    widths = [int(x) for x in args.widths.split(",")]
    pool_idx = [int(x) for x in args.pool_idx.split(",") if x.strip()!=""]

    # num_classes is unused in SSL heads; keep placeholder
    model = AdaptiveCNN(in_ch=3, num_classes=10, widths=widths, pooling_indices=pool_idx).to(device)

    cfg = {
        "objective": args.objective,
        "inner_epochs": args.epochs,
        "patience_depth": args.patience_depth,
        "patience_width": args.patience_width,
        "ex_k": args.ex_k,
        "delta": args.delta,
        "max_width": args.max_width,
    }

    best_val = VARIANT_FUNCS[args.variant](model, cfg, train_loader, device)
    print(f"[DONE] Variant={args.variant} Objective={args.objective} BestProxyLoss={best_val:.4f}")
    # Save final snapshot
    torch.save({"state_dict": model.state_dict(), "widths": model.widths,
                "variant": args.variant, "objective": args.objective},
               f"adp_ssl_{args.variant}_{args.objective}.pt")

if __name__ == "__main__":
    main()

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from adp_ssl_cnn_model import AdaptiveCNN, VARIANT_FUNCS

def get_dataset(name, train=True):
    name = name.lower()
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])
    if name == "cifar10":
        ds = datasets.CIFAR10(root="./data", train=train, download=True, transform=tf_train if train else tf_eval)
    elif name == "cifar100":
        ds = datasets.CIFAR100(root="./data", train=train, download=True, transform=tf_train if train else tf_eval)
    elif name == "stl10":
        split = "train" if train else "test"
        ds = datasets.STL10(root="./data", split=split, download=True, transform=tf_train if train else tf_eval)
    else:
        raise ValueError("Unknown dataset")
    return ds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","cifar100","stl10"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3, help="inner epochs per proposal")
    p.add_argument("--objective", type=str, default="rotation",
                   choices=["rotation","jigsaw","context","masked_ae","colorization","whitening","rot_jigsaw",
                            "temporal_order","optical_flow","exemplar","predictive_coding"])
    p.add_argument("--variant", type=str, default="wd",
                   choices=["wd","dw","alt_d","alt_w","depth_only","width_only"])
    p.add_argument("--widths", type=str, default="32,32,64")
    p.add_argument("--pool_idx", type=str, default="1", help="0-based comma-separated indices for MaxPool")
    p.add_argument("--ex_k", type=int, default=16)
    p.add_argument("--patience_depth", type=int, default=2)
    p.add_argument("--patience_width", type=int, default=2)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--max_width", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = get_dataset(args.dataset, train=True)
    val_frac = 0.1
    n_val = int(len(train_ds)*val_frac)
    n_train = len(train_ds)-n_val
    train_ds, val_ds = random_split(train_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    widths = [int(x) for x in args.widths.split(",")]
    pool_idx = [int(x) for x in args.pool_idx.split(",") if x.strip()!=""]

    model = AdaptiveCNN(in_ch=3, num_classes=10, widths=widths, pooling_indices=pool_idx).to(device)

    cfg = {
        "objective": args.objective,
        "inner_epochs": args.epochs,
        "patience_depth": args.patience_depth,
        "patience_width": args.patience_width,
        "ex_k": args.ex_k,
        "delta": args.delta,
        "max_width": args.max_width,
    }

    best_val = VARIANT_FUNCS[args.variant](model, cfg, train_loader, device)
    print(f"[DONE] Variant={args.variant} Objective={args.objective} BestProxyLoss={best_val:.4f}")
    torch.save({"state_dict": model.state_dict(), "widths": model.widths,
                "variant": args.variant, "objective": args.objective},
               f"adp_ssl_{args.variant}_{args.objective}.pt")

if __name__ == "__main__":
    main()
