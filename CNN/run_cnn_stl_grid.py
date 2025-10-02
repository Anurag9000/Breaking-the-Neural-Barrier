#!/usr/bin/env python3
"""
run_cnn_stl_grid.py
-------------------
Full grid sweep over (depth, width):

- depth: 1..10 (step 1)
- width: 10..250 (step 10)

For each (w, d):
  * build ConvNetSTL
  * train with early stopping (independent run)
  * record:
      - neurons = width * (depth + 1)  (via stl_total_neurons)
      - best_val_loss, best_val_acc, test_acc
  * save per-run epoch plots (loss/acc) using the existing train() utilities

After all runs:
  * save CSV of all rows
  * make two *combined* scatter plots (all runs on one figure):
        1) neurons vs best_val_acc      (markers labeled "(w,d)")
        2) neurons vs best_val_loss     (y in log scale, markers labeled "(w,d)")
  * save a JSON summary with config and metrics

Usage (examples):
-----------------
# CIFAR-100, default settings, CUDA if available
python run_cnn_stl_grid.py --dataset cifar100 --download

# CIFAR-10, CPU only, smaller epochs just to smoke test
python run_cnn_stl_grid.py --dataset cifar10 --device cpu --epochs 5 --patience 2 --download

Notes:
- This script imports the *functions* (make_loaders, train, evaluate) from the user's existing
  `run_cnn_stl.py` to avoid duplication and ensure consistency.
- Per-run plots are written under results_dir/<prefix>_w{w}_d{d}_loss/acc.png by the training utility.
- Combined scatter plots + CSV go into results_dir/.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch

# Reuse the canonical model/metric utilities
from CNN_STL_grid import ConvNetSTL, stl_total_neurons

# Reuse the training/data pipeline (to keep behavior identical to single-run script)
from run_cnn_stl import make_loaders, train, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-sweep ConvNetSTL over (width, depth) and aggregate plots/CSV")

    # Data
    p.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-split", type=int, default=5000)
    p.add_argument("--download", action="store_true")

    # Training
    p.add_argument("--epochs", type=int, default=50)
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


def _parse_pooling(arg: str) -> list:
    s = str(arg or "").strip()
    if not s:
        return []
    out = sorted({int(x) for x in s.split(",") if x.strip() != ""})
    for v in out:
        if v < 0:
            raise ValueError("pool indices must be >= 0")
    return out


def _ensure_dir(p: str):
    Path(p or ".").mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    # Seed / device
    torch.manual_seed(args.seed)
    device = _device_auto() if args.device == "auto" else torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # Prepare output dir
    results_dir = Path(getattr(args, "results_dir", args.results_dir if hasattr(args, "results_dir") else "results_grid"))
    _ensure_dir(results_dir.as_posix())

    # Data loaders (built once, reused across runs)
    train_loader, val_loader, test_loader, num_classes = make_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        download=args.download,
    )

    # Grid
    depths = list(range(args.depth_min, args.depth_max + 1, args.depth_step))
    widths = list(range(args.width_min, args.width_max + 1, args.width_step))
    pool_indices = _parse_pooling(args.pool)

    rows: List[dict] = []

    # Sweep all combinations
    total = len(depths) * len(widths)
    k = 0
    for d in depths:
        for w in widths:
            k += 1
            print(f"[Run {k}/{total}] width={w} depth={d}")

            model = ConvNetSTL(
                input_channels=3,
                num_classes=num_classes,
                width=w,
                depth=d,
                pooling_indices=pool_indices,
            )

            # Per-run plot files will be placed under the results dir with unique prefixes
            plot_prefix = f"ConvNetSTL_w{w}_d{d}"
            stats = train(
                model, train_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                patience=args.patience, log_every=args.log_every,
                plot_dir=results_dir.as_posix(), plot_prefix=plot_prefix, plot_interval_s=1e9
            )

            # Evaluate the best checkpoint on test set
            if os.path.exists("ConvNetSTL_best.pth"):
                model.load_state_dict(torch.load("ConvNetSTL_best.pth", map_location=device))
            test_loss, test_acc = evaluate(model.to(device), test_loader, device)

            neurons = int(stl_total_neurons(model))
            row = {
                "width": int(w),
                "depth": int(d),
                "neurons": neurons,
                "best_val_loss": float(stats["best_val_loss"]),
                "best_val_acc": float(stats["best_val_acc"]),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "loss_plot_path": stats.get("loss_plot"),
                "acc_plot_path": stats.get("acc_plot"),
            }
            rows.append(row)

            # Persist incremental JSON
            with open(results_dir / "partial_results.json", "w") as f:
                json.dump(rows, f, indent=2)

    # Save CSV
    csv_path = results_dir / args.csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "width", "depth", "neurons",
            "best_val_loss", "best_val_acc",
            "test_loss", "test_acc",
            "loss_plot_path", "acc_plot_path",
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Save final JSON
    json_path = results_dir / args.json_name
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    # Combined scatter plots
    xs = [r["neurons"] for r in rows]
    ys_acc = [r["best_val_acc"] for r in rows]
    ys_loss = [max(r["best_val_loss"], 1e-12) for r in rows]  # avoid log(0)
    labels = [f"({r['width']},{r['depth']})" for r in rows]

    # Accuracy vs neurons
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys_acc)
    for x, y, lab in zip(xs, ys_acc, labels):
        plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    plt.xlabel("Total neurons (width * (depth + 1))")
    plt.ylabel("Best validation accuracy")
    plt.title("Accuracy vs Neurons (all configurations)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    acc_plot_path = results_dir / args.acc_plot
    plt.savefig(acc_plot_path.as_posix())
    plt.close()

    # Loss vs neurons (log y)
    plt.figure(figsize=(8, 6))
    plt.semilogy(xs, ys_loss, linestyle="", marker="o")
    for x, y, lab in zip(xs, ys_loss, labels):
        plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    plt.xlabel("Total neurons (width * (depth + 1))")
    plt.ylabel("Best validation loss (log scale)")
    plt.title("Loss vs Neurons (all configurations)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    loss_plot_path = results_dir / args.loss_plot
    plt.savefig(loss_plot_path.as_posix())
    plt.close()

    print(f"\n[GRID DONE] CSV  : {csv_path}")
    print(f"[GRID DONE] JSON : {json_path}")
    print(f"[GRID DONE] PLOT : {acc_plot_path}")
    print(f"[GRID DONE] PLOT : {loss_plot_path}")


if __name__ == "__main__":
    main()
