#!/usr/bin/env python3
"""
Verbose grid runner for ConvNetSTL with epoch-level logging.

Changes vs your previous grid script:
- Adds robust logging (console + file) with timestamps.
- Writes a global log: logs/stl.log and a per-run log: logs/stl_w{w}_d{d}.log.
- Prints per-epoch train/val metrics, best value tracking, patience counter, LR, time.
- Logs when early stopping triggers and when switching to the next model/config.
- Still produces CSV + JSON + combined scatter plots across ALL configs.

NOTE: We keep using your dataset loader from run_cnn_stl.make_loaders to preserve
transforms/splits; however we implement our own training loop here to guarantee
rich logging and deterministic early-stopping behaviour.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from CNN_STL import ConvNetSTL, stl_total_neurons  # your canonical model
from run_cnn_stl import make_loaders  # reuse your data pipeline

# -----------------------------
# Utility: logging configuration
# -----------------------------

def setup_logging(results_dir: Path, run_tag: str) -> logging.Logger:
    log_dir = results_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stl_grid")
    logger.setLevel(logging.INFO)

    # Clear old handlers (so multiple runs don't duplicate logs)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Global file log (append)
    global_f = logging.FileHandler(log_dir / "stl.log", mode="a", encoding="utf-8")
    global_f.setFormatter(fmt)
    logger.addHandler(global_f)

    # Per-run file log (overwrite each run)
    run_f = logging.FileHandler(log_dir / f"stl_{run_tag}.log", mode="w", encoding="utf-8")
    run_f.setFormatter(fmt)
    logger.addHandler(run_f)

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Make Python stdout unbuffered-like
    sys.stdout.reconfigure(line_buffering=True)
    return logger


# -----------------------------
# Metrics helpers
# -----------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == y).sum().item()
    return correct / max(1, y.numel())


@dataclass
class TrainStats:
    best_val_loss: float
    best_val_acc: float
    best_epoch: int
    epochs_ran: int
    acc_plot: str
    loss_plot: str


# -----------------------------
# Training with detailed logging
# -----------------------------

def train_verbose(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    log_every: int = 1,
    plot_dir: str = None,
    plot_prefix: str = "run",
) -> TrainStats:
    logger = logging.getLogger("stl_grid")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = math.inf
    best_val_acc = 0.0
    best_epoch = -1
    bad = 0

    hist_neurons: List[Tuple[int, float]] = []  # (total_neurons, best_val_loss)

    # For simple plotting of loss/acc over epochs
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_accs: List[float] = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * yb.size(0)
            seen += yb.size(0)

        train_loss = running / max(1, seen)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        v_loss = 0.0
        v_seen = 0
        v_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * yb.size(0)
                v_seen += yb.size(0)
                v_correct += (logits.argmax(1) == yb).sum().item()

        val_loss = v_loss / max(1, v_seen)
        val_acc = v_correct / max(1, v_seen)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        improved = val_loss < best_val_loss - 1e-12
        if improved:
            best_val_loss = val_loss
            best_val_acc = max(best_val_acc, val_acc)
            best_epoch = epoch
            bad = 0
        else:
            bad += 1

        # record neurons vs best loss
        hist_neurons.append((stl_total_neurons(model), best_val_loss))

        # Logging line per epoch
        if epoch % max(1, log_every) == 0:
            logger.info(
                "[epoch %d/%d] train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | best_val=%.4f@%d | bad=%d/%d | lr=%.2e | dt=%.1fs",
                epoch, epochs, train_loss, val_loss, val_acc, best_val_loss, best_epoch, bad, patience, optimizer.param_groups[0]['lr'], time.time() - t0,
            )

        # Early stopping
        if bad >= patience:
            logger.info("Early stopping: no improvement for %d epochs (best=%.4f @ epoch %d)", bad, best_val_loss, best_epoch)
            break

    # Finalize plots for THIS run
    plot_dir_p = Path(plot_dir or ".")
    plot_dir_p.mkdir(parents=True, exist_ok=True)

    # Loss curve
    loss_path = plot_dir_p / f"{plot_prefix}_loss.png"
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path.as_posix())
    plt.close()

    # Acc curve
    acc_path = plot_dir_p / f"{plot_prefix}_acc.png"
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(val_accs) + 1), val_accs, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("val_acc")
    plt.title("Validation Accuracy")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(acc_path.as_posix())
    plt.close()

    logger.info("Run finished in %.1fs | best_val_loss=%.4f @epoch %d | best_val_acc=%.4f",
                time.time() - start_time, best_val_loss, best_epoch, best_val_acc)

    return TrainStats(
        best_val_loss=float(best_val_loss),
        best_val_acc=float(best_val_acc),
        best_epoch=int(best_epoch),
        epochs_ran=len(val_losses),
        acc_plot=acc_path.as_posix(),
        loss_plot=loss_path.as_posix(),
    )


# -----------------------------
# Evaluation helper (test set)
# -----------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval().to(device)
    loss_sum, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * yb.size(0)
            n += yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
    return loss_sum / max(1, n), correct / max(1, n)


# -----------------------------
# Argparse
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-sweep ConvNetSTL with verbose logging")

    # Data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])  # default to cifar10 per your request
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)  # Windows safe
    p.add_argument("--val-split", type=int, default=5000)
    p.add_argument("--download", action="store_true")

    # Training
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--log-every", type=int, default=1)

    # Device / misc
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    # Grid ranges
    p.add_argument("--depth-min", type=int, default=1)
    p.add_argument("--depth-max", type=int, default=10)
    p.add_argument("--depth-step", type=int, default=1)
    p.add_argument("--width-min", type=int, default=10)
    p.add_argument("--width-max", type=int, default=250)
    p.add_argument("--width-step", type=int, default=10)

    # Pooling (applied to all runs)
    p.add_argument("--pool", type=str, default="1,3",
                   help="comma-separated 1-based block indices for MaxPool2d, e.g., '1,3'; empty for none")

    # Outputs
    p.add_argument("--results-dir", type=str, default="results_grid")
    p.add_argument("--csv-name", type=str, default="grid_results.csv")
    p.add_argument("--json-name", type=str, default="grid_results.json")
    p.add_argument("--acc-plot", type=str, default="neurons_vs_acc.png")
    p.add_argument("--loss-plot", type=str, default="neurons_vs_loss.png")

    return p.parse_args()


def _device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_pooling(arg: str) -> List[int]:
    s = str(arg or "").strip()
    if not s:
        return []
    out = sorted({int(x) for x in s.split(",") if x.strip() != ""})
    for v in out:
        if v < 0:
            raise ValueError("pool indices must be >= 0")
    return out


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

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

# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = _device_auto() if args.device == "auto" else torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    results_dir = Path(args.results_dir)
    _ensure_dir(results_dir)

    # Data (once)
    train_loader, val_loader, test_loader, num_classes = make_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        download=args.download,
    )

    depths = list(range(args.depth_min, args.depth_max + 1, args.depth_step))
    widths = list(range(args.width_min, args.width_max + 1, args.width_step))
    pool_indices = _parse_pooling(args.pool)

    rows: List[Dict] = []

    total = len(depths) * len(widths)
    run_idx = 0
    for d in depths:
        for w in widths:
            run_idx += 1
            run_tag = f"w{w}_d{d}"
            logger = setup_logging(results_dir, run_tag)
            logger.info("======== [Run %d/%d] width=%d depth=%d ========", run_idx, total, w, d)
            logger.info("Device=%s | dataset=%s | batch=%d | max_epochs=%d | patience=%d | lr=%.2e | wd=%.1e",
                        device, args.dataset, args.batch_size, args.epochs, args.patience, args.lr, args.weight_decay)

            model = ConvNetSTL(
                input_channels=3,
                num_classes=num_classes,
                width=w,
                depth=d,
                pooling_indices=pool_indices,
            )
            params = sum(p.numel() for p in model.parameters())
            neurons = int(stl_total_neurons(model))
            logger.info("Model params=%d | neurons=%d | pooling=%s", params, neurons, pool_indices)

            # Train with verbose per-epoch logging
            stats = train_verbose(
                model,
                train_loader,
                val_loader,
                device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                log_every=args.log_every,
                plot_dir=results_dir.as_posix(),
                plot_prefix=f"ConvNetSTL_w{w}_d{d}",
            )

            # Evaluate best model as-is (we didn't save checkpoints here; using final weights)
            test_loss, test_acc = evaluate(model.to(device), test_loader, device)
            logger.info("Test: loss=%.4f acc=%.4f", test_loss, test_acc)

            row = {
                "width": int(w),
                "depth": int(d),
                "neurons": neurons,
                "best_val_loss": float(stats.best_val_loss),
                "best_val_acc": float(stats.best_val_acc),
                "best_epoch": int(stats.best_epoch),
                "epochs_ran": int(stats.epochs_ran),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "loss_plot_path": stats.loss_plot,
                "acc_plot_path": stats.acc_plot,
            }
            rows.append(row)

            # Incremental dump
            with open(results_dir / "partial_results.json", "w") as f:
                json.dump(rows, f, indent=2)

            logger.info("======== [End Run %d/%d] width=%d depth=%d | best_val=%.4f | test_acc=%.4f ========\n",
                        run_idx, total, w, d, stats.best_val_loss, test_acc)
            _save_combined_scatter(
                rows,
                results_dir,
                args.acc_plot,   # e.g., "acc_vs_neurons.png"
                args.loss_plot,  # e.g., "loss_vs_neurons.png"
            )
 

    # Persist CSV/JSON
    csv_path = results_dir / args.csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "width", "depth", "neurons",
            "best_val_loss", "best_val_acc", "best_epoch", "epochs_ran",
            "test_loss", "test_acc",
            "loss_plot_path", "acc_plot_path",
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    json_path = results_dir / args.json_name
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    _save_combined_scatter(rows, results_dir, args.acc_plot, args.loss_plot)
    acc_plot_path = results_dir / args.acc_plot
    loss_plot_path = results_dir / args.loss_plot
    logger = logging.getLogger("stl_grid")
    logger.info("CSV: %s", csv_path)
    logger.info("JSON: %s", json_path)
    logger.info("PLOT: %s", acc_plot_path)
    logger.info("PLOT: %s", loss_plot_path)


if __name__ == "__main__":
    # Ensure immediate flush of prints
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()
