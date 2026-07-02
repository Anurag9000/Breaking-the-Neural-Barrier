import copy
import csv
import datetime as _dt
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger


def make_loaders(
    dataset: str,
    data_dir: str,
    img_size: Tuple[int, int],
    batch_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
):
    tf = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    name = dataset.lower()
    if name == "cifar10":
        ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf)
        in_ch = 3
        num_classes = 10
    elif name == "cifar100":
        ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=tf)
        in_ch = 3
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use cifar10 or cifar100.")

    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(int(seed))
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    in_dim = int(in_ch * img_size[0] * img_size[1])
    return dl_train, dl_val, in_dim, num_classes


def train_epoch(model: nn.Module, dl, opt, device, *, grad_clip: float) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        total_loss += loss.item() * x.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        n += x.size(0)
    return float(total_loss / max(n, 1)), float(total_correct / max(n, 1))


@torch.no_grad()
def eval_epoch(model: nn.Module, dl, device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        n += x.size(0)
    return float(total_loss / max(n, 1)), float(total_correct / max(n, 1))


def plot_grid_summary(csv_path: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    widths: List[int] = []
    best_val_losses: List[float] = []
    best_val_accs_at_best_loss: List[float] = []
    best_val_accs_any: List[float] = []

    with Path(csv_path).open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            widths.append(int(row["width"]))
            best_val_losses.append(float(row["best_val_loss"]))
            best_val_accs_at_best_loss.append(float(row["val_acc_at_best_loss"]))
            best_val_accs_any.append(float(row["best_val_acc"]))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss vs width
    plt.figure(figsize=(10, 6))
    plt.plot(widths, best_val_losses, marker="o", linewidth=2)
    plt.xlabel("Width")
    plt.ylabel("Best Val Loss (cross-entropy)")
    plt.title("Grid Search: Best Val Loss vs Width")
    plt.grid(True, alpha=0.3, linestyle="--")
    if min(best_val_losses) > 0:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir / "best_val_loss_vs_width.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy vs width
    plt.figure(figsize=(10, 6))
    plt.plot(widths, best_val_accs_at_best_loss, marker="s", linewidth=2, label="Val acc @ best val loss")
    plt.plot(widths, best_val_accs_any, marker="^", linewidth=2, label="Best val acc (any epoch)")
    plt.xlabel("Width")
    plt.ylabel("Validation Accuracy")
    plt.title("Grid Search: Validation Accuracy vs Width")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_acc_vs_width.png", dpi=150, bbox_inches="tight")
    plt.close()


def _train_one_width(
    *,
    width: int,
    depth: int,
    dataset: str,
    data_dir: str,
    img_size: Tuple[int, int],
    seed: int,
    batch_size: int,
    num_workers: int,
    val_split: float,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    max_epochs: int,
    patience: int,
    out_dir: Path,
    device_str: str,
) -> Path:
    """
    Train a single (depth, width) MLP classifier with early stopping and log to out_dir.
    Returns the path to width_summary.csv.
    """
    torch.manual_seed(int(seed))
    device = torch.device(device_str)

    dl_train, dl_val, in_dim, num_classes = make_loaders(
        dataset, data_dir, img_size, batch_size, val_split, seed, num_workers
    )

    models_dir = Path(__file__).resolve().parents[1] / "Models"
    sys.path.append(str(models_dir))
    from mlp_cls_stl import MLPClassifier  # type: ignore

    hidden = [int(width)] * int(depth)
    model = MLPClassifier(in_dim, hidden_widths=hidden, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    logger = ContinuousLogger(out_dir, "mlp_cls_stl", "grid")
    logger.log_console(f"Device: {device}")
    logger.log_console(f"[GRID] width={width} depth={depth} hidden={hidden}")

    best_val_loss = float("inf")
    best_val_acc_at_best_loss = 0.0
    best_epoch_loss = 0
    best_state = None

    best_val_acc_any = 0.0
    best_epoch_acc = 0
    es_counter = 0

    try:
        for epoch in range(1, int(max_epochs) + 1):
            tr_loss, tr_acc = train_epoch(model, dl_train, opt, device, grad_clip=float(grad_clip))
            va_loss, va_acc = eval_epoch(model, dl_val, device)

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_val_acc_at_best_loss = va_acc
                best_epoch_loss = epoch
                best_state = copy.deepcopy(model.state_dict())
                es_counter = 0
                improved = True
            else:
                es_counter += 1
                improved = False

            if va_acc > best_val_acc_any:
                best_val_acc_any = va_acc
                best_epoch_acc = epoch

            logger.log_epoch_stats(
                {
                    "epoch": epoch,
                    "width": int(width),
                    "depth": int(depth),
                    "neurons": int(sum(hidden) + num_classes),
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "best_val": best_val_loss,
                    "es_counter": es_counter,
                    "improved": improved,
                    "grid": True,
                }
            )

            if es_counter >= int(patience):
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        summary_csv = out_dir / "width_summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "width",
                    "depth",
                    "best_val_loss",
                    "val_acc_at_best_loss",
                    "best_val_acc",
                    "epoch_at_best_loss",
                    "epoch_at_best_acc",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "width": int(width),
                    "depth": int(depth),
                    "best_val_loss": float(best_val_loss),
                    "val_acc_at_best_loss": float(best_val_acc_at_best_loss),
                    "best_val_acc": float(best_val_acc_any),
                    "epoch_at_best_loss": int(best_epoch_loss),
                    "epoch_at_best_acc": int(best_epoch_acc),
                }
            )

        logger.log_console(
            f"[GRID] done width={width} best_val_loss={best_val_loss:.6f} "
            f"val_acc@best_loss={best_val_acc_at_best_loss:.4f} best_val_acc={best_val_acc_any:.4f}"
        )
        return summary_csv
    finally:
        logger.close()


def _merge_width_summaries(summary_paths: List[Path], out_csv: Path) -> None:
    rows: List[Dict[str, str]] = []
    for p in summary_paths:
        with Path(p).open("r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)

    rows.sort(key=lambda r: int(r["width"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "width",
                "depth",
                "best_val_loss",
                "val_acc_at_best_loss",
                "best_val_acc",
                "epoch_at_best_loss",
                "epoch_at_best_acc",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Grid search: MLP classifier width sweep (fixed depth)")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--img-size", type=int, nargs=2, default=[28, 28])

    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--width-start", type=int, default=2)
    p.add_argument("--width-step", type=int, default=4)
    p.add_argument("--width-end", type=int, default=256)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-split", type=float, default=0.1)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--patience", type=int, default=10)

    p.add_argument("--results-dir", type=str, default="results_grid/mlp_cls")
    p.add_argument("--parallel", type=int, default=0, help="Run widths in parallel processes (0=off, N>0=workers)")
    p.add_argument(
        "--devices",
        type=str,
        default="",
        help="Comma-separated devices for parallel mode, e.g. 'cuda:0,cuda:1' or 'cpu'. If empty: uses single device.",
    )
    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(default_device)

    run_name = (
        f"{args.dataset}_grid_depth{args.depth}_w{args.width_start}-{args.width_end}_step{args.width_step}_"
        f"{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.results_dir) / run_name
    logger = ContinuousLogger(out_dir, "mlp_cls_stl", "grid")

    summary_csv = out_dir / "grid_summary.csv"
    summary_fields = [
        "width",
        "depth",
        "best_val_loss",
        "val_acc_at_best_loss",
        "best_val_acc",
        "epoch_at_best_loss",
        "epoch_at_best_acc",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()

    widths = list(range(int(args.width_start), int(args.width_end) + 1, int(args.width_step)))
    logger.log_console(f"Device (default): {device}")
    logger.log_console(f"Dataset: {args.dataset} img_size={tuple(args.img_size)}")
    logger.log_console(f"Grid: depth={args.depth} widths={widths[:5]}...{widths[-5:]} (n={len(widths)})")
    logger.log_console(f"Train: batch_size={args.batch_size} lr={args.lr} wd={args.weight_decay} patience={args.patience} max_epochs={args.max_epochs}")

    try:
        if int(args.parallel) > 0:
            import concurrent.futures as cf

            device_list = [d.strip() for d in str(args.devices).split(",") if d.strip()]
            if not device_list:
                device_list = [default_device]
            logger.log_console(f"Parallel: workers={int(args.parallel)} devices={device_list}")

            per_width_dirs = {w: (out_dir / f"width_{w:04d}") for w in widths}
            for d in per_width_dirs.values():
                d.mkdir(parents=True, exist_ok=True)

            # Avoid CPU oversubscription when spawning multiple processes.
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")

            summary_paths: List[Path] = []
            with cf.ProcessPoolExecutor(max_workers=int(args.parallel)) as ex:
                futs = []
                for i, width in enumerate(widths):
                    device_str = device_list[i % len(device_list)]
                    futs.append(
                        ex.submit(
                            _train_one_width,
                            width=int(width),
                            depth=int(args.depth),
                            dataset=str(args.dataset),
                            data_dir=str(args.data_dir),
                            img_size=tuple(args.img_size),
                            seed=int(args.seed),
                            batch_size=int(args.batch_size),
                            num_workers=int(args.num_workers),
                            val_split=float(args.val_split),
                            lr=float(args.lr),
                            weight_decay=float(args.weight_decay),
                            grad_clip=float(args.grad_clip),
                            max_epochs=int(args.max_epochs),
                            patience=int(args.patience),
                            out_dir=per_width_dirs[int(width)],
                            device_str=str(device_str),
                        )
                    )

                for fut in cf.as_completed(futs):
                    summary_paths.append(Path(fut.result()))

            _merge_width_summaries(summary_paths, summary_csv)
        else:
            dl_train, dl_val, in_dim, num_classes = make_loaders(
                args.dataset,
                args.data_dir,
                tuple(args.img_size),
                int(args.batch_size),
                float(args.val_split),
                int(args.seed),
                int(args.num_workers),
            )

            # Import baseline model from the Models directory.
            models_dir = Path(__file__).resolve().parents[1] / "Models"
            sys.path.append(str(models_dir))
            from mlp_cls_stl import MLPClassifier  # type: ignore

            for width in widths:
                hidden = [int(width)] * int(args.depth)
                model = MLPClassifier(in_dim, hidden_widths=hidden, num_classes=num_classes).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

                best_val_loss = float("inf")
                best_val_acc_at_best_loss = 0.0
                best_epoch_loss = 0
                best_state = None

                best_val_acc_any = 0.0
                best_epoch_acc = 0

                logger.log_console(f"[GRID] width={width} depth={args.depth} hidden={hidden}")
                es_counter = 0

                for epoch in range(1, int(args.max_epochs) + 1):
                    tr_loss, tr_acc = train_epoch(model, dl_train, opt, device, grad_clip=float(args.grad_clip))
                    va_loss, va_acc = eval_epoch(model, dl_val, device)

                    # Track best-by-loss (for early stopping)
                    if va_loss < best_val_loss:
                        best_val_loss = va_loss
                        best_val_acc_at_best_loss = va_acc
                        best_epoch_loss = epoch
                        best_state = copy.deepcopy(model.state_dict())
                        es_counter = 0
                        improved = True
                    else:
                        es_counter += 1
                        improved = False

                    # Track best-by-accuracy (for reporting)
                    if va_acc > best_val_acc_any:
                        best_val_acc_any = va_acc
                        best_epoch_acc = epoch

                    logger.log_epoch_stats(
                        {
                            "epoch": epoch,
                            "width": int(width),
                            "depth": int(args.depth),
                            "neurons": int(sum(hidden) + num_classes),
                            "train_loss": tr_loss,
                            "train_acc": tr_acc,
                            "val_loss": va_loss,
                            "val_acc": va_acc,
                            "best_val": best_val_loss,
                            "es_counter": es_counter,
                            "improved": improved,
                            "grid": True,
                        }
                    )

                    if es_counter >= int(args.patience):
                        break

                if best_state is not None:
                    model.load_state_dict(best_state)

                row = {
                    "width": int(width),
                    "depth": int(args.depth),
                    "best_val_loss": float(best_val_loss),
                    "val_acc_at_best_loss": float(best_val_acc_at_best_loss),
                    "best_val_acc": float(best_val_acc_any),
                    "epoch_at_best_loss": int(best_epoch_loss),
                    "epoch_at_best_acc": int(best_epoch_acc),
                }

                with summary_csv.open("a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=summary_fields)
                    w.writerow(row)

                logger.log_console(
                    f"[GRID] done width={width} best_val_loss={best_val_loss:.6f} "
                    f"val_acc@best_loss={best_val_acc_at_best_loss:.4f} best_val_acc={best_val_acc_any:.4f}"
                )

        plot_grid_summary(summary_csv, out_dir)
        logger.log_console(f"Saved summary: {summary_csv}")
        logger.log_console(f"Saved plots: {out_dir / 'best_val_loss_vs_width.png'} and {out_dir / 'val_acc_vs_width.png'}")
    finally:
        logger.close()


if __name__ == "__main__":
    try:
        import os as _os, sys as _sys
        if _os.name == "posix" and _sys.platform.startswith("linux"):
            import ctypes as _ctypes
            _ctypes.CDLL("libc.so.6", use_errno=True).mlockall(3)
        elif _os.name == "nt":
            import ctypes as _ctypes
            _ctypes.windll.kernel32.SetProcessWorkingSetSize(_ctypes.windll.kernel32.GetCurrentProcess(), -1, -1)
    except Exception:
        pass
    main()
