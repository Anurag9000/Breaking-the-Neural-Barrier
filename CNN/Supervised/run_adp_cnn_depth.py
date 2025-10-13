# train_adp_cnn_depth_cifar10.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from adp_cnn_depth import Config as Cfg, ADP_CNN_Depth

# --------- CIFAR-10 stats (canonical) ---------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_loaders_cifar10(
    data_root="data",
    batch_size=128,
    val_split=5000,   # 10% of 50k train set by default
    num_workers=0,
    pin_memory=True,
    download=False,   # do NOT re-download
):
    """
    Build train/val/test loaders for full CIFAR-10 (all 10 classes).
      - train: RandomCrop+Flip + Normalize
      - val/test: only Normalize
    Uses TWO dataset instances for train/val to avoid transform leakage.
    Deterministic split with a fixed seed.
    """
    train_tfms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tfms = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Separate dataset objects so transforms don't clash
    train_ds_aug = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=train_tfms
    )
    train_ds_eval = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=eval_tfms
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=eval_tfms
    )

    total_train = len(train_ds_aug)  # 50_000
    # Cap/adjust val_split safely
    if val_split >= total_train:
        val_split = max(1, int(0.2 * total_train))
    n_train = total_train - val_split

    # Deterministic split
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(total_train, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    # Subsets with distinct (aug vs eval) transforms
    train_subset = Subset(train_ds_aug, train_idx)
    val_subset   = Subset(train_ds_eval, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Tiny budget sanity run")
    ap.add_argument("--data-root", type=str, default="data", help="Where CIFAR-10 is stored")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-split", type=int, default=5000)  # 10% of train set
    ap.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing")
    args = ap.parse_args()

    # CIFAR-10 settings:
    # - init width = 10 (if model uses widths)
    # - trials_width = 10 and trials_depth = 10 (if supported)
    # - expansion factor ex_k = 10 per loop (if used)
    params = dict(
        delta=0.0,
        trials_depth=100,
        trials_width=100,
        patience=100,
        max_epochs=1000000,
        init_widths=[5],       # <<< init width = 10
        num_classes=10,         # full CIFAR-10
        pooling_indices=[0],
        lr=1e-3,
        weight_decay=1e-2,
        ex_k=5,                # <<< per-loop expansion factor = 10
        max_neurons=1_000_000,
        max_depth=10000,
        max_width=10000,
    )
    if args.smoke:
        params.update(dict(max_epochs=5, patience=2))

    dev = device_auto()
    train_loader, val_loader, test_loader = make_loaders_cifar10(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    cfg = Cfg(**params)
    model = ADP_CNN_Depth(cfg, device=dev)

    model.fit(train_loader, val_loader)
    test_loss, test_acc = model.evaluate(test_loader)
    print(f"[ADP_CNN_Depth-CIFAR10] test_loss={test_loss:.4f} acc={test_acc:.4f}")

    torch.save(model.model.state_dict(), "ADP_CNN_Depth_CIFAR10.pth")
    print("Saved: ADP_CNN_Depth_CIFAR10.pth")

if __name__ == "__main__":
    main()
