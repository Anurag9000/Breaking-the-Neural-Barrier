"""
Grid STL runner for ADPResNet.

Sweeps over width/depth combinations, trains one STL model per config,
and writes a CSV/JSON plus combined Neurons-vs-{Loss,Accuracy} plots.
"""
from __future__ import annotations

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

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from CNN.ADP_ResNet.adp_resnet_backbone import ADPResNet, make_adp_resnet, estimate_neurons
from CNN.ADP_ResNet.run_adp_resnet import make_loaders
from utils.cutmix import cutmix_batch


def _device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def setup_logging(results_dir: Path, run_tag: str) -> logging.Logger:
    log_dir = results_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("resnet_stl_grid")
    logger.setLevel(logging.INFO)

    # Clear old handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    global_f = logging.FileHandler(log_dir / "resnet_stl.log", mode="a", encoding="utf-8")
    global_f.setFormatter(fmt)
    logger.addHandler(global_f)

    run_f = logging.FileHandler(log_dir / f"resnet_stl_{run_tag}.log", mode="w", encoding="utf-8")
    run_f.setFormatter(fmt)
    logger.addHandler(run_f)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    sys.stdout.reconfigure(line_buffering=True)
    return logger


@dataclass
class TrainStats:
    best_val_loss: float
    best_val_acc: float
    best_epoch: int
    epochs_ran: int


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def train_verbose(
    model: ADPResNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    log_every: int,
    cutmix_p: float,
    cutmix_alpha: float,
) -> TrainStats:
    logger = logging.getLogger("resnet_stl_grid")

    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = math.inf
    best_val_acc = 0.0
    best_epoch = -1
    bad = 0

    gen = torch.Generator(device=device)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running, seen = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            xb_mix, targets = cutmix_batch(
                xb,
                yb,
                alpha=cutmix_alpha,
                p=cutmix_p,
                generator=gen,
            )

            opt.zero_grad(set_to_none=True)
            logits = model(xb_mix)
            if targets is None:
                loss = crit(logits, yb)
            else:
                y1, y2, lam = targets
                loss = lam * crit(logits, y1) + (1.0 - lam) * crit(logits, y2)

            loss.backward()
            opt.step()

            running += loss.item() * xb.size(0)
            seen += xb.size(0)

        train_loss = running / max(1, seen)

        model.eval()
        v_loss, v_acc = evaluate(model, val_loader, device)

        improved = v_loss < best_val_loss - 0
        if improved:
            best_val_loss = float(v_loss)
            best_val_acc = float(v_acc)
            best_epoch = epoch
            bad = 0
        else:
            bad += 1

        if epoch % max(1, log_every) == 0:
            logger.info(
                "[epoch %d/%d] train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | best_val=%.4f@%d | bad=%d/%d | dt=%.1fs",
                epoch,
                epochs,
                train_loss,
                v_loss,
                v_acc,
                best_val_loss,
                best_epoch,
                bad,
                patience,
                time.time() - t0,
            )

        if bad >= patience:
            logger.info(
                "Early stopping: no improvement for %d epochs (best=%.4f @ epoch %d)",
                bad,
                best_val_loss,
                best_epoch,
            )
            break

    return TrainStats(
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        epochs_ran=epoch,
    )


def _save_combined_scatter(rows: List[Dict], results_dir: Path, acc_name: str, loss_name: str) -> None:
    if not rows:
        return

    from collections import defaultdict

    by_depth = defaultdict(list)
    for r in rows:
        by_depth[int(r["depth"])].append(r)

    depths_sorted = sorted(by_depth.keys())
    cmap = plt.get_cmap("tab20")
    depth_to_color = {d: cmap(i % cmap.N) for i, d in enumerate(depths_sorted)}

    # Accuracy vs neurons
    plt.figure(figsize=(8, 6))
    for d in depths_sorted:
        grp = by_depth[d]
        xs = [g["neurons"] for g in grp]
        ys = [g["best_val_acc"] for g in grp]
        labs = [f"({g['width']},{g['depth']})" for g in grp]
        plt.scatter(xs, ys, label=f"depth={d}", s=28, alpha=0.9, edgecolors="none",
                    c=[depth_to_color[d]] * len(xs))
        for x, y, lab in zip(xs, ys, labs):
            plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    plt.xlabel("Estimated neurons (width × (depth + 1))")
    plt.ylabel("Best validation accuracy")
    plt.title("ADPResNet STL: Accuracy vs Neurons")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(title="Depth", fontsize=8, title_fontsize=9, ncol=2, frameon=True)
    plt.tight_layout()
    _ensure_dir(results_dir)
    plt.savefig((results_dir / acc_name).as_posix())
    plt.close()

    # Loss vs neurons (log-y)
    plt.figure(figsize=(8, 6))
    for d in depths_sorted:
        grp = by_depth[d]
        xs = [g["neurons"] for g in grp]
        ys = [max(g["best_val_loss"], 1e-12) for g in grp]
        labs = [f"({g['width']},{g['depth']})" for g in grp]
        plt.semilogy(xs, ys, linestyle="", marker="o", markersize=4,
                     c=depth_to_color[d], label=f"depth={d}")
        for x, y, lab in zip(xs, ys, labs):
            plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    plt.xlabel("Estimated neurons (width × (depth + 1))")
    plt.ylabel("Best validation loss (log scale)")
    plt.title("ADPResNet STL: Loss vs Neurons")
    plt.grid(True, ls="--", alpha=0.5, which="both")
    plt.legend(title="Depth", fontsize=8, title_fontsize=9, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig((results_dir / loss_name).as_posix())
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid STL runner for ADPResNet on CIFAR.")

    # Data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--download", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--num-classes", type=int, default=10,
                   help="Use only the first N classes (labels 0..N-1). For CIFAR-10, 2–10 are valid.")

    # Grid
    p.add_argument("--depth-min", type=int, default=1)
    p.add_argument("--depth-max", type=int, default=3)
    p.add_argument("--depth-step", type=int, default=1)
    p.add_argument("--width-min", type=int, default=8)
    p.add_argument("--width-max", type=int, default=64)
    p.add_argument("--width-step", type=int, default=8)

    # Optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--log-every", type=int, default=1)

    # CutMix
    p.add_argument("--cutmix-p", type=float, default=0.0)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)

    # Device / results
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--results-dir", type=str, default="results_resnet_stl_grid")
    p.add_argument("--csv-name", type=str, default="grid_results.csv")
    p.add_argument("--json-name", type=str, default="grid_results.json")
    p.add_argument("--acc-plot", type=str, default="neurons_vs_acc.png")
    p.add_argument("--loss-plot", type=str, default="neurons_vs_loss.png")
    p.add_argument("--seed", type=int, default=123)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = _device_auto() if args.device == "auto" else torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    results_dir = Path(args.results_dir)
    _ensure_dir(results_dir)

    train_loader, val_loader, num_classes = make_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        use_augment=not args.no_augment,
        num_classes_limit=args.num_classes,
    )

    depths = list(range(args.depth_min, args.depth_max + 1, args.depth_step))
    widths = list(range(args.width_min, args.width_max + 1, args.width_step))

    rows: List[Dict] = []
    total_runs = len(depths) * len(widths)
    run_idx = 0

    for d in depths:
        for w in widths:
            run_idx += 1
            run_tag = f"w{w}_d{d}"
            logger = setup_logging(results_dir, run_tag)
            logger.info(
                "======== [Run %d/%d] width=%d depth=%d ========",
                run_idx,
                total_runs,
                w,
                d,
            )
            logger.info(
                "Device=%s | dataset=%s | batch=%d | max_epochs=%d | patience=%d | lr=%.2e | wd=%.1e | cutmix_p=%.2f",
                device,
                args.dataset,
                args.batch_size,
                args.epochs,
                args.patience,
                args.lr,
                args.weight_decay,
                args.cutmix_p,
            )

            model = make_adp_resnet(
                input_channels=3,
                num_classes=num_classes,
                width=w,
                depth=d,
            )
            params = sum(p.numel() for p in model.parameters())
            neurons = int(estimate_neurons(w, d))
            logger.info("Model params=%d | neurons=%d", params, neurons)

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
                cutmix_p=args.cutmix_p,
                cutmix_alpha=args.cutmix_alpha,
            )

            test_loss, test_acc = evaluate(model.to(device), val_loader, device)
            logger.info("Val-as-test: loss=%.4f acc=%.4f", test_loss, test_acc)

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
            }
            rows.append(row)

            with open(results_dir / "partial_results.json", "w") as f:
                json.dump(rows, f, indent=2)

            logger.info(
                "======== [End Run %d/%d] width=%d depth=%d | best_val=%.4f | test_acc=%.4f ========\n",
                run_idx,
                total_runs,
                w,
                d,
                stats.best_val_loss,
                test_acc,
            )

            _save_combined_scatter(rows, results_dir, args.acc_plot, args.loss_plot)

    csv_path = results_dir / args.csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "width",
                "depth",
                "neurons",
                "best_val_loss",
                "best_val_acc",
                "best_epoch",
                "epochs_ran",
                "test_loss",
                "test_acc",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    json_path = results_dir / args.json_name
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    logger = logging.getLogger("resnet_stl_grid")
    logger.info("CSV: %s", csv_path)
    logger.info("JSON: %s", json_path)
    logger.info("PLOT acc: %s", results_dir / args.acc_plot)
    logger.info("PLOT loss: %s", results_dir / args.loss_plot)


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()
