from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize final metrics across repeats for tabular ablation runs.")
    p.add_argument(
        "--results-root",
        default="MLPS/tabular/shared/dae_dnn/results/stl_ablation_parameter_matched_gpu_serial",
        help="Root directory containing per-candidate repeat outputs.",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path. Defaults to <results-root>/repeat_final_metrics_summary.csv",
    )
    return p.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_completed_candidates(results_root: Path) -> Iterable[Tuple[Dict[str, Any], Path, List[Dict[str, Any]]]]:
    for csv_path in results_root.glob("**/training_stats.csv"):
        state_path = csv_path.parent / "candidate_state.json"
        if not state_path.exists() or csv_path.stat().st_size == 0:
            continue
        try:
            state = read_json(state_path)
        except Exception:
            continue
        if not bool(state.get("completed", False)):
            continue
        try:
            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows:
            continue
        yield state, csv_path, rows


def final_numeric(row: Dict[str, Any], key: str) -> Optional[float]:
    raw = row.get(key)
    if raw in (None, "", "na", "NA"):
        return None
    try:
        return float(raw)
    except Exception:
        return None


def summarize(values: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not values:
        return None, None, None, None
    mean = stats.fmean(values)
    var = stats.pvariance(values) if len(values) > 1 else 0.0
    std = math.sqrt(var)
    cv = (std / abs(mean) * 100.0) if mean not in (0, None) else None
    return mean, var, std, cv


def to_scientific_parts(value: float) -> Tuple[str, str]:
    n = int(round(float(value)))
    exp = 0
    mant = float(n)
    while mant >= 10:
        mant /= 10
        exp += 1
    return f"{mant:.1f}", str(exp)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_csv = Path(args.output_csv) if args.output_csv else results_root / "repeat_final_metrics_summary.csv"

    grouped: Dict[Tuple[str, Optional[int], Optional[int]], List[Dict[str, Any]]] = defaultdict(list)

    for state, csv_path, rows in iter_completed_candidates(results_root):
        last = rows[-1]
        architecture = state.get("architecture") or []
        depth = len(architecture) if isinstance(architecture, list) and architecture else None
        width_match = re.search(r"_w(\d+)", csv_path.parent.as_posix())
        width = int(width_match.group(1)) if width_match else None
        grouped[(str(state.get("task")), depth, width)].append(
            {
                "train_loss": final_numeric(last, "train_loss"),
                "val_loss": final_numeric(last, "val_loss"),
                "train_acc": final_numeric(last, "train_acc"),
                "val_acc": final_numeric(last, "val_acc"),
            }
        )

    fieldnames = [
        "task",
        "depth",
        "width",
        "repeats_done",
        "train_loss_mean",
        "train_loss_var",
        "train_loss_std",
        "train_loss_cv_pct",
        "val_loss_mean",
        "val_loss_var",
        "val_loss_std",
        "val_loss_cv_pct",
        "loss_spread_pct",
        "train_acc_mean",
        "train_acc_var",
        "train_acc_std",
        "train_acc_cv_pct",
        "val_acc_mean",
        "val_acc_var",
        "val_acc_std",
        "val_acc_cv_pct",
    ]
    rows_out: List[Dict[str, Any]] = []
    for (task, depth, width), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2] or 0)):
        def collect(key: str) -> List[float]:
            return [v[key] for v in vals if v[key] is not None]

        tl = summarize(collect("train_loss"))
        vl = summarize(collect("val_loss"))
        ta = summarize(collect("train_acc"))
        va = summarize(collect("val_acc"))
        final_val_losses = collect("val_loss")
        loss_spread_pct = None
        if final_val_losses:
            best = min(final_val_losses)
            worst = max(final_val_losses)
            loss_spread_pct = ((worst - best) / best * 100.0) if best != 0 else None

        rows_out.append(
            {
                "task": task,
                "depth": depth,
                "width": width,
                "repeats_done": len(vals),
                "train_loss_mean": tl[0],
                "train_loss_var": tl[1],
                "train_loss_std": tl[2],
                "train_loss_cv_pct": tl[3],
                "val_loss_mean": vl[0],
                "val_loss_var": vl[1],
                "val_loss_std": vl[2],
                "val_loss_cv_pct": vl[3],
                "loss_spread_pct": loss_spread_pct,
                "train_acc_mean": ta[0],
                "train_acc_var": ta[1],
                "train_acc_std": ta[2],
                "train_acc_cv_pct": ta[3],
                "val_acc_mean": va[0],
                "val_acc_var": va[1],
                "val_acc_std": va[2],
                "val_acc_cv_pct": va[3],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"wrote {output_csv} with {len(rows_out)} rows")


def write_scientific_parameter_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "depth", "width", "params_mantissa", "params_power10"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            mantissa, power = to_scientific_parts(float(row["params"]))
            writer.writerow(
                {
                    "task": row["task"],
                    "depth": row["depth"],
                    "width": row["width"],
                    "params_mantissa": mantissa,
                    "params_power10": power,
                }
            )


if __name__ == "__main__":
    main()
