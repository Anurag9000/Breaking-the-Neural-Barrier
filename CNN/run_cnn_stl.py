# train_adp_cnn_width_cifar10.py
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from adp_cnn_width import Config as Cfg, ADP_CNN_Width

# --------- CIFAR-10 stats ---------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_loaders_cifar10(
    data_root="data",
    batch_size=128,
    val_split=5000,   # 10% of the 50k train set by default
    num_workers=0,
    pin_memory=True,
    download=False,   # do NOT re-download unless explicitly asked
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
    save_prefix: str = "ConvNetSTL_sweep",
):
    \"\"\"
    Sweep one hyperparameter (width or depth) while keeping the other fixed.
    For example:
      fixed='width', base_width=10, base_depth=100, ex_k=10
      -> depths = [10, 20, ..., 100], width fixed at 10

      fixed='depth', base_depth=10, base_width=100, ex_k=10
      -> widths = [10, 20, ..., 100], depth fixed at 10
    For each config: train with early stopping, record (neurons, best_val_loss), plot semilogy.
    \"\"\"
    assert fixed in {"width", "depth"}, "fixed must be 'width' or 'depth'"
    # Determine sweep values
    if fixed == "width":
        # sweep depth from ex_k to base_depth inclusive by ex_k
        sweep_vals = list(range(max(ex_k, 1), max(base_depth, ex_k) + 1, ex_k))
    else:
        # sweep width from ex_k to base_width inclusive by ex_k
        sweep_vals = list(range(max(ex_k, 1), max(base_width, ex_k) + 1, ex_k))

    # Create loaders once (shared across runs)
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
            input_channels=3, num_classes=num_classes, width=w, depth=d, pooling_indices=[]
        )
    
    # Sweep mode (if requested)
    if args.sweep:
        if (args.sweep_fixed is None) or (args.ex_k is None):
            raise SystemExit("--sweep requires --sweep-fixed {width|depth} and --ex-k <step>")
        # We use args.width/args.depth only as the upper bounds and the fixed value respectively
        sweep_json = sweep_train(
            base_width=args.width, base_depth=args.depth, fixed=args.sweep_fixed, ex_k=args.ex_k,
            dataset=args.dataset, data_root=args.data_root, batch_size=args.batch_size,
            num_workers=args.num_workers, val_split=args.val_split, download=args.download,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience,
            log_every=args.log_every, device=device, plot_path=args.sweep_plot, save_prefix=args.sweep_prefix
        )
        print(f"[SWEEP] Saved: {sweep_json['plot_path']} and {args.sweep_prefix}_sweep.json")
        return

    # Train
        stats = train(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            patience=patience, log_every=log_every,
            plot_dir="results_stl", plot_prefix=f"{save_prefix}_w{w}_d{d}", plot_interval_s=1e9  # disable mid-epoch autosave
        )

        # Get neurons (parity with CNN_STL metric)
        try:
            from CNN_STL import stl_total_neurons
            neurons = int(stl_total_neurons(model))
        except Exception:
            neurons = int(d * w)  # fallback

        xs_neurons.append(neurons)
        ys_best_val.append(float(stats["best_val_loss"]))

        print(f\"[Sweep {i}/{len(sweep_vals)}] width={w} depth={d} neurons={neurons} "
              f"best_val_loss={stats['best_val_loss']:.6f}\")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Tiny budget sanity run")
    ap.add_argument("--data-root", type=str, default="data", help="Where CIFAR-10 is stored")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-split", type=int, default=5000)
    ap.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing")
    args = ap.parse_args()

    params = dict(
        delta=0.0,
        trials_width=10,
        trials_depth=10,     # kept at 10 if your width-search variant also iterates depth
        patience=10,
        max_epochs=100000,
        init_widths=[10],    # <<< init width = 10
        num_classes=10,      # full CIFAR-10
        pooling_indices=[0],
        lr=1e-3,
        weight_decay=1e-2,
        ex_k=10,             # <<< per-loop expansion factor = 10 (if used)
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
        download=args.download,
    )

    cfg = Cfg(**params)
    model = ADP_CNN_Width(cfg, device=dev)

    model.fit(train_loader, val_loader)
    test_loss, test_acc = model.evaluate(test_loader)
    print(f"[ADP_CNN_Width-CIFAR10] test_loss={test_loss:.4f} acc={test_acc:.4f}")

    torch.save(model.model.state_dict(), "ADP_CNN_Width_CIFAR10.pth")
    print("Saved: ADP_CNN_Width_CIFAR10.pth")

if __name__ == "__main__":
    main()
