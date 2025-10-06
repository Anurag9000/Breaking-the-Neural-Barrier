import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from adp_cnn_den_depth_only import Config as Cfg, ADP_CNN_DepthOnly

# --------- Real CIFAR-10 (32x32 RGB) ---------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def make_loaders_cifar10(
    data_root="data",
    batch_size=128,
    num_workers=0,
    pin_memory=True,
):
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

    # Two train-set views: one with aug for training, one with eval tfms for validation
    train_ds_aug = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=train_tfms
    )
    train_ds_eval = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=eval_tfms
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=eval_tfms
    )

    # Deterministic 90/10 split of the 50,000 training images
    total_train = len(train_ds_aug)  # 50_000
    val_split = int(0.1 * total_train)  # 5_000
    n_train = total_train - val_split    # 45_000

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(total_train, generator=g).tolist()
    train_idxs = perm[:n_train]
    val_idxs   = perm[n_train:]

    # Important: use aug set for train and eval set for val
    train_subset = torch.utils.data.Subset(train_ds_aug, train_idxs)
    val_subset   = torch.utils.data.Subset(train_ds_eval, val_idxs)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0)
    )
    return train_loader, val_loader, test_loader

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Fast sanity run with tiny budgets")
    ap.add_argument("--data-root", type=str, default="data", help="Where CIFAR-10 is stored")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    # Defaults for CIFAR-10 (10 classes).
    # As requested:
    #   - init width = 10
    #   - trials_width = 10 (if applicable in your Config)
    #   - trials_depth = 10
    #   - expansion factor ex_k = 10 (if used by your model loop)
    params = dict(
        delta=0.0,
        trials_depth=100,
        trials_width=100,          # if supported by your Config/Model
        patience=100,
        max_epochs=100000,
        init_widths=[10],         # <<< init width = 10
        num_classes=10,
        pooling_indices=[0],
        lr=1e-3,
        weight_decay=1e-2,
        ex_k=10,                  # <<< per-loop expansion factor = 10 (if used)
        max_neurons=1_000_000,
    )
    if args.smoke:
        params.update(dict(max_epochs=5, patience=2))

    dev = device_auto()
    train_loader, val_loader, test_loader = make_loaders_cifar10(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    cfg = Cfg(**params)
    model = ADP_CNN_DepthOnly(cfg, device=dev)

    model.fit(train_loader, val_loader)
    test_loss, test_acc = model.evaluate(test_loader)
    print(f"[ADP_CNN_DepthOnly-CIFAR10] test_loss={test_loss:.4f} acc={test_acc:.4f}")

    os.makedirs(".", exist_ok=True)
    torch.save(model.model.state_dict(), "ADP_CNN_DepthOnly_CIFAR10.pth")
    print("Saved: ADP_CNN_DepthOnly_CIFAR10.pth")

if __name__ == "__main__":
    main()
