from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import run_goliath as rg
import run_stl_ablation as stl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge parameter-band STL ablation roots into a canonical combined root.")
    p.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help="One or more completed band roots produced by run_stl_ablation_parallel.py.",
    )
    p.add_argument("--output-root", required=True, help="Canonical combined STL ablation root.")
    p.add_argument("--tasks", nargs="+", default=list(stl.DEFAULT_TASKS))
    p.add_argument("--overwrite", action="store_true", help="Overwrite an existing output root if it already exists.")
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    data = rg.load_json_if_exists(path)
    return data if isinstance(data, dict) else {}


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def discover_param_band(root: Path, summary: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    band = summary.get("param_band")
    if isinstance(band, list) and len(band) == 2:
        try:
            return int(band[0]), int(band[1])
        except Exception:
            pass
    meta = load_json(root / "comparison_summary.json")
    band = meta.get("param_band")
    if isinstance(band, list) and len(band) == 2:
        try:
            return int(band[0]), int(band[1])
        except Exception:
            return None
    return None


def parameter_count_from_candidate(checkpoint_best: str) -> int:
    candidate_dir = Path(checkpoint_best).parent
    metadata = rg.load_json_if_exists(candidate_dir / "metadata.json") or {}
    model_cfg = metadata.get("model", {})
    model = rg.make_model(
        int(model_cfg.get("in_dim", 0)),
        [int(v) for v in model_cfg.get("hidden_widths", [])],
        int(model_cfg.get("out_dim", 0)),
        bool(model_cfg.get("use_bn", False)),
    )
    return int(sum(p.numel() for p in model.parameters()))


def merge_task(task_name: str, output_root: Path, input_roots: Sequence[Path]) -> Dict[str, Any]:
    task_root = output_root / task_name
    task_root.mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, Any]] = []
    merged_comparisons: List[Dict[str, Any]] = []
    merged_runs: List[Dict[str, Any]] = []
    merged_band_roots: List[str] = []
    source_task_summaries: List[str] = []
    best_ablation: Optional[Dict[str, Any]] = None
    repeat_count: Optional[int] = None
    parameter_budget_target: Optional[int] = None
    parameter_matched: Optional[bool] = None
    gpu_samples: List[int] = []

    for band_root in input_roots:
        summary_path = band_root / task_name / "ablation_summary.json"
        csv_path = band_root / task_name / "ablation_summary.csv"
        if not summary_path.exists() or not csv_path.exists():
            continue
        summary = load_json(summary_path)
        if not summary.get("ablation_stl_runs"):
            continue

        band_rows = load_csv_rows(csv_path)
        for row in band_rows:
            row = dict(row)
            row["source_root"] = str(band_root)
            merged_rows.append(row)

        band_band = discover_param_band(band_root, summary)
        if band_band is not None:
            merged_band_roots.append(f"{band_root}::depth_{band_band[0]:02d}_{band_band[1]:02d}")
        else:
            merged_band_roots.append(str(band_root))

        source_task_summaries.append(str(summary.get("source_task_summary")))
        repeat_count = int(summary.get("repeat_count", repeat_count or 0)) if repeat_count is None else repeat_count
        parameter_budget_target = (
            int(summary.get("parameter_budget_target"))
            if parameter_budget_target is None and summary.get("parameter_budget_target") is not None
            else parameter_budget_target
        )
        parameter_matched = bool(summary.get("parameter_matched")) if parameter_matched is None else parameter_matched
        gpu_samples.extend(int(v) for v in summary.get("gpu_vram_samples_mib", []) if isinstance(v, (int, float)))

        for entry in summary.get("ablation_stl_runs", []):
            entry = dict(entry)
            entry["source_root"] = str(band_root)
            merged_runs.append(entry)
            if best_ablation is None or float(entry.get("best_val", float("inf"))) < float(best_ablation.get("best_val", float("inf"))):
                best_ablation = entry

        merged_comparisons.extend(summary.get("comparisons", []))

    if not merged_rows:
        return {
            "task": task_name,
            "merged_from": [],
            "ablation_stl_runs": [],
            "comparisons": [],
            "best_ablation": None,
        }

    fieldnames = [
        "task",
        "repeat",
        "row_type",
        "phase",
        "architecture",
        "parameter_count",
        "best_val",
        "best_epoch",
        "final_epoch",
        "test_loss",
        "test_acc",
        "source_root",
    ]
    rg.write_csv(task_root / "ablation_summary.csv", merged_rows, fieldnames=fieldnames)

    gpu_vram_avg_mib = float(sum(gpu_samples) / max(len(gpu_samples), 1)) if gpu_samples else None
    merged_summary = {
        "task": task_name,
        "merged_from": merged_band_roots,
        "source_task_summaries": source_task_summaries,
        "parameter_matched": parameter_matched,
        "parameter_budget_target": parameter_budget_target,
        "gpu_vram_samples_mib": gpu_samples,
        "gpu_vram_avg_mib": gpu_vram_avg_mib,
        "ablation_stl_runs": merged_runs,
        "comparisons": merged_comparisons,
        "best_ablation": best_ablation,
        "repeat_count": repeat_count,
        "architecture_count": len({tuple(int(v) for v in stl.parse_architecture(str(row["architecture"]))) for row in merged_rows}),
    }
    rg.write_json(task_root / "ablation_summary.json", merged_summary)

    plot_path = stl.plot_task_ablation(task_root, task_name, merged_rows)
    return {
        "task": task_name,
        "merged_from": merged_band_roots,
        "best_ablation": best_ablation,
        "plot_path": str(plot_path),
        "ablation_stl_runs": merged_runs,
        "comparisons": merged_comparisons,
    }


def main() -> None:
    args = parse_args()
    input_roots = [Path(root) for root in args.input_roots]
    output_root = Path(args.output_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    logger = rg.ContinuousLogger(output_root, "stl_ablation_merge", "stl_ablation_merge")
    logger.log_console(f"Output root: {output_root}")
    logger.log_console(f"Input roots: {[str(root) for root in input_roots]}")
    logger.log_console(f"Tasks: {args.tasks}")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    task_reports: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    try:
        for task_name in args.tasks:
            report = merge_task(task_name, output_root, input_roots)
            if report.get("best_ablation") is None:
                continue
            task_reports.append(report)
            comparison_rows.extend(report.get("comparisons", []))

        if comparison_rows:
            rg.write_csv(
                output_root / "comparison_summary.csv",
                comparison_rows,
                fieldnames=[
                    "task",
                    "repeat",
                    "ablation_phase",
                    "ablation_architecture",
                    "ablation_parameter_count",
                    "ablation_best_val",
                    "reference_kind",
                    "reference_phase",
                    "reference_architecture",
                    "reference_best_val",
                    "winner",
                    "winner_value",
                ],
            )

        rg.write_json(
            output_root / "comparison_summary.json",
            {
                "tasks": args.tasks,
                "input_roots": [str(root) for root in input_roots],
                "reports": task_reports,
            },
        )
    finally:
        rg.cleanup_runtime()
        logger.close()


if __name__ == "__main__":
    main()
