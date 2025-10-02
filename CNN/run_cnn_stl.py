
import argparse
import json
import os
import time
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# Import the STL CNN
from CNN_STL import ConvNetSTL  # requires CNN_STL.py in the same folder

# -------------------- CIFAR stats --------------------
CIFAR10_MEAN, CIFAR10_STD   = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(
    dataset: str,
    data_root: str = "data",
    batch_size: int = 128,
    val_split: int = 5000,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train/val/test loaders for CIFAR-10 or CIFAR-100 with standard aug.
    Returns loaders and num_classes.
    """
    dataset = dataset.lower()
    if dataset not in {"cifar10", "cifar100"}:
        raise ValueError("dataset must be 'cifar10' or 'cifar100'")

    if dataset == "cifar10":
        MEAN, STD = CIFAR10_MEAN, CIFAR10_STD
        ds_train = torchvision.datasets.CIFAR10
        ds_test  = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        MEAN, STD = CIFAR100_MEAN, CIFAR100_STD
        ds_train = torchvision.datasets.CIFAR100
        ds_test  = torchvision.datasets.CIFAR100
        num_classes = 100

    train_tfms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    eval_tfms = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    train_ds_aug = ds_train(root=data_root, train=True, download=download, transform=train_tfms)
    train_ds_eval = ds_train(root=data_root, train=True, download=False, transform=eval_tfms)
    test_ds = ds_test(root=data_root, train=False, download=False, transform=eval_tfms)

    total_train = len(train_ds_aug)
    if val_split >= total_train:
        val_split = max(1, int(0.2 * total_train))  # keep ~20% if requested too large
    n_train = total_train - val_split

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(total_train, generator=g).tolist()
    train_idx_subset = perm[:n_train]
    val_idx_subset   = perm[n_train:]

    train_ds = Subset(train_ds_aug, train_idx_subset)
    val_ds   = Subset(train_ds_eval, val_idx_subset)

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
    return train_loader, val_loader, test_loader, num_classes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    log_every: int = 1,
) -> dict:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_acc = 0.0
    bad = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            n += y.size(0)

        train_loss = running / max(1, n)
        val_loss, val_acc = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if epoch % log_every == 0:
            print(f"[epoch {epoch}/{epochs}] "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        improved = val_loss < best_val - 1e-6  # tiny margin
        if improved:
            best_val = val_loss
            best_acc = val_acc
            bad = 0
            torch.save(model.state_dict(), "ConvNetSTL_best.pth")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    return {
        "best_val_loss": best_val,
        "best_val_acc": best_acc,
        "history": history,
    }


def parse_pooling(arg: str) -> List[int]:
    """
    Parse a comma-separated list of integers like '1,3'.
    Returns a sorted list; empty string -> []
    """
    if arg.strip() == "":
        return []
    out = sorted({int(x) for x in arg.split(",")})
    for v in out:
        if v < 0:
            raise ValueError("pool indices must be >= 0")
    return out


def main():
    parser = argparse.ArgumentParser(description="Run ConvNetSTL on CIFAR-10/100")
    # Data
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=int, default=5000)
    parser.add_argument("--download", action="store_true")

    # Model (STL) hyperparameters exposed here
    parser.add_argument("--width", type=int, default=64, help="channels per block")
    parser.add_argument("--depth", type=int, default=4, help="number of ConvBNReLU blocks (>=1)")
    parser.add_argument("--pool", type=parse_pooling, default="1,3",
                        help="comma-separated block indices after which to apply 2x2 MaxPool, e.g. '1,3'")

    # Optimization
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-prefix", type=str, default="ConvNetSTL")
    parser.add_argument("--log-every", type=int, default=1)

    args = parser.parse_args()

    # Seed & device
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = device_auto()
    else:
        device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    # Load data
    train_loader, val_loader, test_loader, num_classes = make_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        download=args.download,
    )

    # Build model
    model = ConvNetSTL(
        input_channels=3,
        num_classes=num_classes,
        width=args.width,
        depth=args.depth,
        pooling_indices=args.pool,
    )

    # Train
    t0 = time.time()
    stats = train(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, log_every=args.log_every
    )
    train_time = time.time() - t0

    # Evaluate best checkpoint on test
    if os.path.exists("ConvNetSTL_best.pth"):
        model.load_state_dict(torch.load("ConvNetSTL_best.pth", map_location=device))
    test_loss, test_acc = evaluate(model.to(device), test_loader, device)

    # Save final artifacts
    out_prefix = args.save_prefix
    final_path = f"{out_prefix}.pth"
    torch.save(model.state_dict(), final_path)

    report = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "width": args.width,
        "depth": args.depth,
        "pooling_indices": args.pool,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "best_val_loss": stats["best_val_loss"],
        "best_val_acc": stats["best_val_acc"],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_time_sec": train_time,
        "checkpoint_best": "ConvNetSTL_best.pth" if os.path.exists("ConvNetSTL_best.pth") else None,
        "final_model": final_path,
    }
    with open(f"{out_prefix}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved final weights to: {final_path}")
    print(f"Saved report to      : {out_prefix}_report.json")
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
