
import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, List, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

import matplotlib.pyplot as plt

# Import the STL CNN + neuron counter
from CNN_STL import ConvNetSTL, stl_total_neurons

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


def _ensure_dir(p: str):
    Path(p or ".").mkdir(parents=True, exist_ok=True)


def _save_semi_log_loss_plot(path: str, epochs: List[int], val_losses: List[float]):
    if not epochs:
        return
    plt.figure(figsize=(6, 4))
    plt.semilogy(epochs, val_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss (log scale)")
    plt.title("Epoch vs Validation Loss")
    plt.grid(True, ls="--", alpha=0.5)
    _ensure_dir(os.path.dirname(path) or ".")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_acc_plot(path: str, epochs: List[int], val_accs: List[float]):
    if not epochs:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_accs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Epoch vs Validation Accuracy")
    plt.grid(True, ls="--", alpha=0.5)
    _ensure_dir(os.path.dirname(path) or ".")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 100,
    log_every: int = 1,
    plot_dir: str = "results_stl",
    plot_prefix: str = "ConvNetSTL",
    plot_interval_s: float = 60.0,
) -> dict:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_acc = 0.0
    bad = 0
    history = []

    # plotting setup (periodic autosave like ADP)
    _ensure_dir(plot_dir)
    loss_plot_path = os.path.join(plot_dir, f"{plot_prefix}_loss.png")
    acc_plot_path  = os.path.join(plot_dir, f"{plot_prefix}_acc.png")
    epochs_hist: List[int] = []
    val_losses_hist: List[float] = []
    val_accs_hist: List[float] = []
    last_plot_t = time.time()

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

        # record for plots
        epochs_hist.append(epoch)
        val_losses_hist.append(max(float(val_loss), 1e-12))
        val_accs_hist.append(float(val_acc))

        # periodic auto-save plot
        now = time.time()
        if now - last_plot_t >= plot_interval_s:
            _save_semi_log_loss_plot(loss_plot_path, epochs_hist, val_losses_hist)
            _save_acc_plot(acc_plot_path, epochs_hist, val_accs_hist)
            print(f"[Plot] updated {loss_plot_path} and {acc_plot_path} (points={len(epochs_hist)})")
            last_plot_t = now

        if epoch % log_every == 0:
            print(f"[epoch {epoch}/{epochs}] "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        improved = val_loss < best_val - 0  # tiny margin
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

    # final save of plots
    _save_semi_log_loss_plot(loss_plot_path, epochs_hist, val_losses_hist)
    _save_acc_plot(acc_plot_path, epochs_hist, val_accs_hist)

    return {
        "best_val_loss": best_val,
        "best_val_acc": best_acc,
        "history": history,
        "loss_plot": loss_plot_path,
        "acc_plot": acc_plot_path,
    }


def parse_pooling(arg: str) -> List[int]:
    """
    Parse a comma-separated list of integers like '1,3'.
    Returns a sorted list; empty string -> []
    """
    if isinstance(arg, list):
        return [int(x) for x in arg]
    if str(arg).strip() == "":
        return []
    out = sorted({int(x) for x in str(arg).split(",")})
    for v in out:
        if v < 0:
            raise ValueError("pool indices must be >= 0")
    return out


def sweep_train(
    base_width: int,
    base_depth: int,
    fixed: str,
    ex_k: int,
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    val_split: int,
    download: bool,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    log_every: int,
    device: torch.device,
    plot_path: str = "results_stl/ConvNetSTL_sweep.png",
    save_prefix: str = "ConvNetSTL",
    sweep_pool: Iterable[int] = (),
):
    """
    Sweep one hyperparameter (width or depth) while keeping the other fixed.
    Example A (fixed width): base_width=10, base_depth=100, ex_k=10 -> depths: 10..100 step 10
    Example B (fixed depth): base_depth=10, base_width=100, ex_k=10 -> widths: 10..100 step 10
    For each config: train with early stopping, record (neurons, best_val_loss), plot semilogy.
    """
    assert fixed in {"width", "depth"}, "fixed must be 'width' or 'depth'"
    if ex_k <= 0:
        raise ValueError("ex_k must be > 0")

    if fixed == "width":
        sweep_vals = list(range(ex_k, max(base_depth, ex_k) + 1, ex_k))
    else:
        sweep_vals = list(range(ex_k, max(base_width, ex_k) + 1, ex_k))

    # Data once
    train_loader, val_loader, test_loader, num_classes = make_loaders(
        dataset=dataset, data_root=data_root, batch_size=batch_size,
        val_split=val_split, num_workers=num_workers, download=download,
    )

    xs_neurons: List[int] = []
    ys_best_val: List[float] = []

    for i, val in enumerate(sweep_vals, 1):
        if fixed == "width":
            w = base_width
            d = val
        else:
            w = val
            d = base_depth

        model = ConvNetSTL(
            input_channels=3, num_classes=num_classes, width=w, depth=d, pooling_indices=sweep_pool
        )

        stats = train(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            patience=patience, log_every=log_every,
            plot_dir="results_stl", plot_prefix=f"{save_prefix}_w{w}_d{d}", plot_interval_s=1e9  # disable mid-epoch autosave
        )

        neurons = int(stl_total_neurons(model))
        xs_neurons.append(neurons)
        ys_best_val.append(float(stats["best_val_loss"]))

        print(f"[Sweep {i}/{len(sweep_vals)}] width={w} depth={d} neurons={neurons} "
              f"best_val_loss={stats['best_val_loss']:.6f}")

    # Plot neurons vs best val loss (semilogy y)
    _ensure_dir(Path(plot_path).parent.as_posix())
    plt.figure(figsize=(6,4))
    plt.semilogy(xs_neurons, ys_best_val, marker="o")
    plt.xlabel("Total neurons (channels sum + head fan-in)")
    plt.ylabel("Best validation loss (log scale)")
    plt.title(f"Sweep ({fixed} fixed)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Save sweep json
    sweep_json = {
        "fixed": fixed,
        "ex_k": ex_k,
        "points": [{"neurons": int(x), "best_val_loss": float(y)} for x, y in zip(xs_neurons, ys_best_val)],
        "plot_path": plot_path,
    }
    with open(f"{save_prefix}_sweep.json", "w") as f:
        json.dump(sweep_json, f, indent=2)

    return sweep_json

def _save_combined_scatter(rows, results_dir: Path, acc_name: str, loss_name: str):
    """
    Make combined Neurons vs {Acc, Loss} figures, with:
    - point label = "(width,depth)"
    - color scheme keyed by 'depth'
    - re-entrant: safe to call after each run
    """
    if not rows:
        return

    # Group by depth for color separation
    by_depth = defaultdict(list)
    for r in rows:
        by_depth[int(r["depth"])].append(r)

    # Stable ordering for legend
    depths_sorted = sorted(by_depth.keys())
    cmap = plt.get_cmap("tab20")  # good categorical palette
    depth_to_color = {d: cmap(i % cmap.N) for i, d in enumerate(depths_sorted)}

    # ---------- Accuracy vs Neurons ----------
    plt.figure(figsize=(8, 6))
    for d in depths_sorted:
        grp = by_depth[d]
        xs = [g["neurons"] for g in grp]
        ys = [g["best_val_acc"] for g in grp]
        labs = [f"({g['width']},{g['depth']})" for g in grp]

        plt.scatter(xs, ys, label=f"depth={d}", s=28, alpha=0.9, edgecolors="none",
                    c=[depth_to_color[d]]*len(xs))
        for x, y, lab in zip(xs, ys, labs):
            plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    plt.xlabel("Total neurons (width × (depth + 1))")
    plt.ylabel("Best validation accuracy")
    plt.title("Accuracy vs Neurons (all configurations)")
    plt.grid(True, ls="--", alpha=0.5)
    # Keep legend compact even for many depths
    plt.legend(title="Depth", fontsize=8, title_fontsize=9, ncol=2, frameon=True)
    plt.tight_layout()
    (Path(results_dir) / acc_name).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig((Path(results_dir) / acc_name).as_posix())
    plt.close()

    # ---------- Loss vs Neurons ----------
    plt.figure(figsize=(8, 6))
    for d in depths_sorted:
        grp = by_depth[d]
        xs = [g["neurons"] for g in grp]
        # guard against exact 0
        ys = [max(g["best_val_loss"], 1e-12) for g in grp]
        labs = [f"({g['width']},{g['depth']})" for g in grp]

        # use semilog-y per your original plot
        plt.semilogy(xs, ys, linestyle="", marker="o", markersize=4,
                     c=depth_to_color[d], label=f"depth={d}")
        for x, y, lab in zip(xs, ys, labs):
            plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    plt.xlabel("Total neurons (width × (depth + 1))")
    plt.ylabel("Best validation loss (log scale)")
    plt.title("Loss vs Neurons (all configurations)")
    plt.grid(True, ls="--", alpha=0.5, which="both")
    plt.legend(title="Depth", fontsize=8, title_fontsize=9, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig((Path(results_dir) / loss_name).as_posix())
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run ConvNetSTL on CIFAR-10/100 (with optional sweep mode)")
    # Data
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=int, default=5000)
    parser.add_argument("--download", action="store_true")

    # Model (STL) hyperparameters exposed here
    parser.add_argument("--width", type=int, default=64, help="channels per block (width)")
    parser.add_argument("--depth", type=int, default=4, help="number of ConvBNReLU blocks (>=1)")
    parser.add_argument("--pool", type=parse_pooling, default="1,3",
                        help="comma-separated block indices after which to apply 2x2 MaxPool, e.g. '1,3'")

    # Optimization
    parser.add_argument("--epochs", type=int, default=5000000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=100)

    # Plots (ADP-style auto-save)
    parser.add_argument("--plot-dir", type=str, default="results_stl")
    parser.add_argument("--plot-prefix", type=str, default="ConvNetSTL")
    parser.add_argument("--plot-interval", type=float, default=60.0, help="seconds between autosaves")

    # Sweep
    parser.add_argument("--sweep", action="store_true", help="Enable sweep mode")
    parser.add_argument("--sweep-fixed", type=str, choices=["width","depth"], default=None,
                        help="Which hyperparameter stays fixed across the sweep")
    parser.add_argument("--ex-k", type=int, default=None, help="Increment step for the changing hyperparameter")
    parser.add_argument("--sweep-plot", type=str, default="results_stl/ConvNetSTL_sweep.png")
    parser.add_argument("--sweep-prefix", type=str, default="ConvNetSTL",
                        help="Prefix for per-run plot files and sweep JSON")
    parser.add_argument("--sweep-pool", type=parse_pooling, default="",
                        help="Pooling indices to use for all configs during sweep (e.g., '1,3'). Empty for none.")

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

    # Sweep mode (if requested)
    if args.sweep:
        if (args.sweep_fixed is None) or (args.ex_k is None):
            raise SystemExit("--sweep requires --sweep-fixed {width|depth} and --ex-k <step>")
        sweep_json = sweep_train(
            base_width=args.width, base_depth=args.depth, fixed=args.sweep_fixed, ex_k=args.ex_k,
            dataset=args.dataset, data_root=args.data_root, batch_size=args.batch_size,
            num_workers=args.num_workers, val_split=args.val_split, download=args.download,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience,
            log_every=args.log_every, device=device, plot_path=args.sweep_plot,
            save_prefix=args.sweep_prefix, sweep_pool=args.sweep_pool
        )
        print(f"[SWEEP] Saved: {sweep_json['plot_path']} and {args.sweep_prefix}_sweep.json")
        return

    # Standard train
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
        patience=args.patience, log_every=args.log_every,
        plot_dir=args.plot_dir, plot_prefix=args.plot_prefix, plot_interval_s=args.plot_interval
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
        "loss_plot": stats.get("loss_plot"),
        "acc_plot": stats.get("acc_plot"),
    }
    with open(f"{out_prefix}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved final weights to: {final_path}")
    print(f"Saved report to      : {out_prefix}_report.json")
    print(f"Saved plots to       : {stats.get('loss_plot')} and {stats.get('acc_plot')}")
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
