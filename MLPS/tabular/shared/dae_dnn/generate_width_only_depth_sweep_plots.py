from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

import run_goliath as rg


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = "MLPS/tabular/shared/dae_dnn/results"
DEFAULT_STL_ABLATION_ROOT = "MLPS/tabular/shared/dae_dnn/results/stl_ablation_all_tasks_d3plus_w64plus"
DEFAULT_OUTPUT_SUBDIR = "analysis/width_only_depth_sweep_loglog"
DEFAULT_TASKS = [
    "classification",
    "autoencoding",
    "generation",
    "denoising",
    "anomaly",
    "simulation",
    "prediction",
]
DEPTHS = [1, 2, 3, 4, 5]


@dataclass
class SeriesPoint:
    family: str
    label: str
    task: str
    architecture: List[int]
    best_val: float
    parameter_count: int
    source: str
    order_key: float

    def to_row(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "label": self.label,
            "task": self.task,
            "architecture": rg.format_architecture_for_report(self.architecture),
            "best_val": float(self.best_val),
            "parameter_count": int(self.parameter_count),
            "source": self.source,
            "order_key": float(self.order_key),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-task width-only depth sweep loss-vs-params plots.")
    p.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--stl-ablation-root", default=DEFAULT_STL_ABLATION_ROOT)
    p.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    return p.parse_args()


def repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_architecture(raw: Any) -> List[int]:
    if isinstance(raw, list):
        return [int(v) for v in raw]
    if isinstance(raw, str):
        value = ast.literal_eval(raw)
        if isinstance(value, int):
            return [int(value)]
        return [int(v) for v in value]
    if isinstance(raw, (int, float)):
        return [int(raw)]
    raise TypeError(f"Unsupported architecture value: {raw!r}")


def parameter_count_from_metadata(metadata_path: Path) -> int:
    meta = read_json(metadata_path)
    model_meta = meta["model"]
    model = rg.make_model(
        int(model_meta["in_dim"]),
        [int(v) for v in model_meta["hidden_widths"]],
        int(model_meta["out_dim"]),
        bool(model_meta["use_bn"]),
    )
    return int(sum(p.numel() for p in model.parameters()))


def collect_adp_depth_points(results_root: Path, task: str, depth: int) -> List[SeriesPoint]:
    phase_root = results_root / f"{task}_ae_width_only_d{depth}" / task / "ae_width_only"
    progress_path = phase_root / "phase_progress.csv"
    if not progress_path.exists():
        return []
    df = pd.read_csv(progress_path)
    points: List[SeriesPoint] = []
    for idx, row in enumerate(df.to_dict(orient="records")):
        architecture = parse_architecture(row["architecture"])
        cand_dir = phase_root / str(row["candidate_dir"])
        metadata_path = cand_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        points.append(
            SeriesPoint(
                family="ADP",
                label=f"ADP d{depth}",
                task=task,
                architecture=architecture,
                best_val=float(row["best_val"]),
                parameter_count=parameter_count_from_metadata(metadata_path),
                source=str(cand_dir / "candidate_state.json"),
                order_key=float(idx),
            )
        )
    return points


def collect_paired_stl_points(results_root: Path, task: str) -> List[SeriesPoint]:
    points: List[SeriesPoint] = []
    for depth in DEPTHS:
        summary_path = results_root / f"{task}_ae_width_only_d{depth}" / task / "task_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        for entry in summary.get("paired_stl_runs", []):
            architecture = [int(v) for v in entry["architecture"]]
            checkpoint_path = repo_path(str(entry["checkpoint_best"]))
            candidate_dir = checkpoint_path.parent
            metadata_path = candidate_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            points.append(
                SeriesPoint(
                    family="STL",
                    label=f"STL refit d{depth}",
                    task=task,
                    architecture=architecture,
                    best_val=float(entry["best_val"]),
                    parameter_count=parameter_count_from_metadata(metadata_path),
                    source=str(summary_path),
                    order_key=float(parameter_count_from_metadata(metadata_path)),
                )
            )
    return points


def collect_ablation_stl_points(stl_ablation_root: Path, task: str) -> List[SeriesPoint]:
    summary_path = stl_ablation_root / task / "ablation_summary.json"
    if not summary_path.exists():
        return []
    summary = read_json(summary_path)
    points: List[SeriesPoint] = []
    for entry in summary.get("ablation_stl_runs", []):
        architecture = [int(v) for v in entry["architecture"]]
        checkpoint_path = repo_path(str(entry["checkpoint_best"]))
        candidate_dir = checkpoint_path.parent
        metadata_path = candidate_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        points.append(
            SeriesPoint(
                family="STL",
                label="STL ablation",
                task=task,
                architecture=architecture,
                best_val=float(entry["best_val"]),
                parameter_count=parameter_count_from_metadata(metadata_path),
                source=str(summary_path),
                order_key=float(parameter_count_from_metadata(metadata_path)),
            )
        )
    return points


def write_task_data(task_dir: Path, task: str, adp_series: Dict[int, List[SeriesPoint]], stl_points: List[SeriesPoint]) -> None:
    rows: List[Dict[str, Any]] = []
    for depth in DEPTHS:
        rows.extend(point.to_row() for point in adp_series.get(depth, []))
    rows.extend(point.to_row() for point in stl_points)
    pd.DataFrame(rows).to_csv(task_dir / f"{task}_depth_sweep_points.csv", index=False)
    rg.write_json(task_dir / f"{task}_depth_sweep_points.json", {"points": rows})


def plot_task(task_dir: Path, task: str, adp_series: Dict[int, List[SeriesPoint]], stl_points: List[SeriesPoint]) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    depth_colors = {
        1: "#d62728",
        2: "#ff7f0e",
        3: "#9467bd",
        4: "#8c564b",
    }
    any_points = False
    for depth in DEPTHS:
        points = adp_series.get(depth, [])
        if not points:
            continue
        any_points = True
        xs = [point.parameter_count for point in points]
        ys = [point.best_val for point in points]
        ax.plot(xs, ys, color=depth_colors[depth], linewidth=2.0, marker="o", markersize=3, label=f"ADP width_only d{depth}")

    if stl_points:
        any_points = True
        stl_points = sorted(stl_points, key=lambda p: (p.parameter_count, p.best_val, p.label))
        xs = [point.parameter_count for point in stl_points]
        ys = [point.best_val for point in stl_points]
        ax.plot(xs, ys, color="green", linewidth=2.0, marker="o", markersize=3, label="STL ablation + refits")

    if not any_points:
        plt.close(fig)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter Count (log scale)")
    ax.set_ylabel("Best Validation Loss (log scale)")
    ax.set_title(f"{task}: width-only depth sweep vs STL")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(task_dir / f"{task}_depth_sweep_loss_vs_params_loglog.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = repo_path(args.results_root)
    stl_ablation_root = repo_path(args.stl_ablation_root)
    output_root = results_root / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    for task in args.tasks:
        task_dir = output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        adp_series = {depth: collect_adp_depth_points(results_root, task, depth) for depth in DEPTHS}
        stl_points = collect_ablation_stl_points(stl_ablation_root, task) + collect_paired_stl_points(results_root, task)
        write_task_data(task_dir, task, adp_series, stl_points)
        plot_task(task_dir, task, adp_series, stl_points)

        for depth in DEPTHS:
            manifest_rows.append(
                {
                    "task": task,
                    "family": "ADP",
                    "label": f"d{depth}",
                    "num_points": len(adp_series.get(depth, [])),
                    "plot_path": str(task_dir / f"{task}_depth_sweep_loss_vs_params_loglog.png"),
                }
            )
        manifest_rows.append(
            {
                "task": task,
                "family": "STL",
                "label": "ablation+refits",
                "num_points": len(stl_points),
                "plot_path": str(task_dir / f"{task}_depth_sweep_loss_vs_params_loglog.png"),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(output_root / "depth_sweep_manifest.csv", index=False)
    rg.write_json(output_root / "depth_sweep_manifest.json", {"entries": manifest_rows})


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
