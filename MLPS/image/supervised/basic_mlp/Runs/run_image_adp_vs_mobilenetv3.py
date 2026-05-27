from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[5]
MLP_ADP = ROOT / "MLPS/image/supervised/basic_mlp/Models/mlp_cls_stl_adp_width_to_depth.py"
CNN_MOBILENETV3 = ROOT / "CNN/Supervised/Runs/run_cnn_mobilenet_v_3.py"


ADP_MODES = ["alt_width", "alt_depth", "width_to_depth", "depth_to_width"]


def _dataset_image_shape(dataset: str) -> List[int]:
    name = dataset.lower()
    if name in {"mnist", "fashionmnist"}:
        return [28, 28]
    return [32, 32]


def _parse_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def _run_cmd(cmd: List[str], cwd: Path, stdout_path: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    stdout_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.stdout + proc.stderr


def _run_mlp_adp(
    dataset: str,
    mode: str,
    args: argparse.Namespace,
    results_dir: Path,
) -> Dict[str, object]:
    out_dir = results_dir / dataset / f"mlp_adp_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(MLP_ADP),
        "--dataset",
        dataset,
        "--data-dir",
        args.data_dir,
        "--results-dir",
        str(out_dir),
        "--adp-mode",
        mode,
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--trials-width",
        str(args.trials_width),
        "--trials-depth",
        str(args.trials_depth),
        "--ex-k",
        str(args.ex_k),
        "--width-stage-margin-patience",
        str(args.width_stage_margin_patience),
        "--width-stage-min-improve-pct",
        str(args.width_stage_min_improve_pct),
        "--max-epochs",
        str(args.max_epochs),
        "--img-size",
        *[str(v) for v in _dataset_image_shape(dataset)],
    ]
    output = _run_cmd(cmd, ROOT, out_dir / "stdout.log")
    best_val = _parse_float(r"best_val_loss=([0-9.]+)", output)
    hidden_match = re.search(r"hidden=\[([^\]]+)\]", output)
    return {
        "dataset": dataset,
        "model": "mlp_adp",
        "mode": mode,
        "best_val_loss": best_val,
        "hidden": hidden_match.group(1) if hidden_match else None,
        "raw_log": str(out_dir / "stdout.log"),
    }


def _run_mobilenetv3(dataset: str, args: argparse.Namespace, results_dir: Path) -> Dict[str, object]:
    out_dir = results_dir / dataset / "mobilenetv3"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "checkpoint.pth"
    cmd = [
        sys.executable,
        str(CNN_MOBILENETV3),
        "--dataset",
        dataset,
        "--data_root",
        args.data_dir,
        "--save_path",
        str(save_path),
        "--version",
        args.version,
        "--width_mult",
        str(args.width_mult),
        "--dropout",
        str(args.dropout),
        "--batch_size",
        str(args.batch_size),
        "--test_batch_size",
        str(args.test_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--max_epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--delta",
        str(args.delta),
        "--grad_clip",
        str(args.grad_clip),
        "--val_frac",
        str(args.val_frac),
        "--seed",
        str(args.seed),
        "--img_size",
        *[str(v) for v in _dataset_image_shape(dataset)],
    ]
    output = _run_cmd(cmd, ROOT, out_dir / "stdout.log")
    best_val = _parse_float(r"best_val_loss=([0-9.]+)", output)
    test_loss = _parse_float(r"test_loss=([0-9.]+)", output)
    test_acc = _parse_float(r"test_acc=([0-9.]+)", output)
    return {
        "dataset": dataset,
        "model": "mobilenetv3",
        "mode": "baseline",
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "raw_log": str(out_dir / "stdout.log"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare image MLP ADP modes against MobileNetV3")
    p.add_argument("--datasets", nargs="+", default=["mnist", "fashionmnist", "cifar10"])
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="results_image_compare")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--trials-width", type=int, default=10)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=1)
    p.add_argument("--width-stage-margin-patience", type=int, default=5)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)

    p.add_argument("--version", choices=["small", "large"], default="small")
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=1e-4)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, object]] = []

    for dataset in args.datasets:
        for mode in ADP_MODES:
            row = _run_mlp_adp(dataset, mode, args, results_dir)
            summary.append(row)
            print(f"[DONE] dataset={dataset} model=mlp_adp mode={mode} best_val_loss={row['best_val_loss']}")

        row = _run_mobilenetv3(dataset, args, results_dir)
        summary.append(row)
        print(f"[DONE] dataset={dataset} model=mobilenetv3 best_val_loss={row['best_val_loss']}")

    summary_csv = results_dir / "comparison_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in summary for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    with (results_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {summary_csv}")


if __name__ == "__main__":
    main()
