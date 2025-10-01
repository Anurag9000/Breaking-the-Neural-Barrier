# train_adp_cnn_widthonly_cifar10.py
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from adp_cnn_den_width_only import Config as Cfg, ADP_CNN_WidthOnly

# --------- CIFAR-10 stats ---------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_loaders_cifar10(
    data_root="data",
    batch_size=128,
    val_split=5000,    # 10% of the 50k training set by default
    num_workers=0,
    pin_memory=True,
    download=False,    # do NOT re-download unless explicitly asked
):
    """
    CIFAR-10 loaders (all 10 classes, full dataset):
      - train: RandomCrop+Flip + Normalize
      - val/test: only Normalize
    Uses separate dataset instances for train/val to avoid transform leakage.
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

    # Independent datasets so transforms don't clash via shared .dataset
    train_ds_aug = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=download, transform=train_tfms
    )
    train_ds_eval = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=eval_tfms
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=eval_tfms
    )

    total_train = len(train_ds_aug)  # 50_000
    if val_split >= total_train:
        # Keep ~20% for validation if requested split is too large
        val_split = max(1, int(0.2 * total_train))
    n_train = total_train - val_split

    # Deterministic split
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(total_train, generator=g).tolist()
    train_idx_subset = perm[:n_train]
    val_idx_subset   = perm[n_train:]

    train_ds = Subset(train_ds_aug,  train_idx_subset)
    val_ds   = Subset(train_ds_eval, val_idx_subset)
    test_ds  = test_ds  # already eval transforms

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
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
    ap.add_argument("--val-split", type=int, default=5000)
    ap.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing")
    args = ap.parse_args()

    # Width-only variant for CIFAR-10:
    # - init width = 10
    # - trials_width = 10 (if supported)
    # - ex_k = 10 per expansion loop (if used)
    params = dict(
        delta=0.0,
        trials_width=10,
        patience=100,
        max_epochs=100000,
        init_widths=[10],       # <<< init width = 10
        num_classes=10,         # full CIFAR-10
        pooling_indices=[0],
        lr=1e-3,
        weight_decay=1e-2,
        ex_k=10,                # <<< per-loop expansion factor = 10
        max_neurons=1_000_000,
    )
    if args.smoke:
        params.update(dict(max_epochs=5, patience=2))

    dev = device_auto()
    train_loader, val_loader, test_loader = make_loaders_cifar10(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        download=args.download,   # never re-download unless asked
    )

    cfg = Cfg(**params)
    model = ADP_CNN_WidthOnly(cfg, device=dev)

    model.fit(train_loader, val_loader)
    test_loss, test_acc = model.evaluate(test_loader)
    print(f"[ADP_CNN_WidthOnly-CIFAR10] test_loss={test_loss:.4f} acc={test_acc:.4f}")

    torch.save(model.model.state_dict(), "ADP_CNN_WidthOnly_CIFAR10.pth")
    print("Saved: ADP_CNN_WidthOnly_CIFAR10.pth")

if __name__ == "__main__":
    main()
