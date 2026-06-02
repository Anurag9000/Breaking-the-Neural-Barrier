from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

import run_goliath as rg


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = "MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu"
DEFAULT_OUTPUT_SUBDIR = "analysis/loss_vs_params_generation_logy"
DEFAULT_TASK = "generation"
DEFAULT_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@dataclass
class SeriesPoint:
    depth: int
    family: str
    label: str
    architecture: List[int]
    best_val: float
    parameter_count: int
    source: str
    order_key: float

    def to_row(self) -> Dict[str, Any]:
        return {
            "depth": int(self.depth),
            "family": self.family,
            "label": self.label,
            "architecture": rg.format_architecture_for_report(self.architecture),
            "best_val": float(self.best_val),
            "parameter_count": int(self.parameter_count),
            "source": self.source,
            "order_key": float(self.order_key),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate generation depth sweep loss-vs-params plot with STL baseline.")
    p.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS)
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
    metadata = read_json(metadata_path)
    model_meta = metadata["model"]
    model = rg.make_model(
        int(model_meta["in_dim"]),
        [int(v) for v in model_meta["hidden_widths"]],
        int(model_meta["out_dim"]),
        bool(model_meta["use_bn"]),
    )
    return int(sum(p.numel() for p in model.parameters()))


def collect_phase_points(results_root: Path, task: str, depth: int, phase_name: str, family: str) -> List[SeriesPoint]:
    phase_root = results_root / f"{task}_d{depth}" / task / phase_name
    progress_path = phase_root / "phase_progress.csv"
    if not progress_path.exists():
        return []

    df = pd.read_csv(progress_path)
    points: List[SeriesPoint] = []
    for row in df.to_dict(orient="records"):
        candidate_dir = phase_root / str(row["candidate_dir"])
        metadata_path = candidate_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        architecture = parse_architecture(row["architecture"])
        points.append(
            SeriesPoint(
                depth=int(depth),
                family=family,
                label=f"{family} d{depth}",
                architecture=architecture,
                best_val=float(row["best_val"]),
                parameter_count=parameter_count_from_metadata(metadata_path),
                source=str(candidate_dir / "candidate_state.json"),
                order_key=float(row.get("candidate_index", len(points))),
            )
        )
    points.sort(key=lambda p: p.order_key)
    return points


def collect_series(results_root: Path, task: str, depths: Sequence[int]) -> tuple[Dict[int, List[SeriesPoint]], List[SeriesPoint]]:
    adp_series: Dict[int, List[SeriesPoint]] = {}
    stl_series: List[SeriesPoint] = []
    for depth in depths:
        adp_series[int(depth)] = collect_phase_points(results_root, task, int(depth), "ae_width_only", "ADP")
        stl_series.extend(collect_phase_points(results_root, task, int(depth), "stl_from_ae_width_only", "STL"))
    stl_series.sort(key=lambda p: (p.parameter_count, p.best_val, p.depth))
    return adp_series, stl_series


def write_manifests(task_dir: Path, adp_series: Dict[int, List[SeriesPoint]], stl_series: Sequence[SeriesPoint]) -> None:
    rows: List[Dict[str, Any]] = []
    for depth in sorted(adp_series):
        rows.extend(point.to_row() for point in adp_series[depth])
    rows.extend(point.to_row() for point in stl_series)
    pd.DataFrame(rows).to_csv(task_dir / "generation_depth_sweep_points.csv", index=False)
    rg.write_json(task_dir / "generation_depth_sweep_points.json", {"points": rows})


def annotate_best_point(ax, point: SeriesPoint, color: str, prefix: str) -> None:
    ax.annotate(
        f"{prefix} {rg.format_architecture_for_report(point.architecture)}\n{point.best_val:.6g}",
        (point.parameter_count, point.best_val),
        textcoords="offset points",
        xytext=(8, -10),
        fontsize=8,
        color=color,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.9},
    )


def plot_task(task_dir: Path, adp_series: Dict[int, List[SeriesPoint]], stl_series: Sequence[SeriesPoint]) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(13, 8))
    cmap = plt.get_cmap("tab10")
    used = False

    for idx, depth in enumerate(sorted(adp_series)):
        points = adp_series[depth]
        if not points:
            continue
        used = True
        color = cmap(idx % 10)
        xs = [p.parameter_count for p in points]
        ys = [p.best_val for p in points]
        ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=3.5, label=f"ADP d{depth}")
        best_point = min(points, key=lambda p: p.best_val)
        annotate_best_point(ax, best_point, color=color, prefix=f"d{depth} best")

    if stl_series:
        used = True
        xs = [p.parameter_count for p in stl_series]
        ys = [p.best_val for p in stl_series]
        ax.plot(xs, ys, color="#2ca02c", linewidth=2.3, linestyle="--", marker="s", markersize=4.0, label="STL refit baseline")
        best_stl = min(stl_series, key=lambda p: p.best_val)
        annotate_best_point(ax, best_stl, color="#2ca02c", prefix="STL best")

    if not used:
        plt.close(fig)
        return None

    ax.set_xlabel("Total number of parameters")
    ax.set_ylabel("Best validation loss (log scale)")
    ax.set_yscale("log")
    ax.set_title("generation: loss vs parameters by depth")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    note = [
        "Each colored line is one generation depth ADP trajectory.",
        "Annotations show the best point on each line as [w1, w2, ...] plus its loss.",
        "The dashed green curve is the STL refit baseline from the saved generation runs.",
    ]
    fig.text(
        0.02,
        0.01,
        "\n".join(note),
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#f8f8f8", "ec": "#cfcfcf", "alpha": 0.96},
    )

    plot_path = task_dir / "generation_loss_vs_params_logy.png"
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    results_root = repo_path(args.results_root)
    output_root = results_root / args.output_subdir
    task_dir = output_root / args.task
    task_dir.mkdir(parents=True, exist_ok=True)

    adp_series, stl_series = collect_series(results_root, args.task, args.depths)
    write_manifests(task_dir, adp_series, stl_series)
    plot_path = plot_task(task_dir, adp_series, stl_series)

    summary = {
        "task": args.task,
        "results_root": str(results_root),
        "plot_path": str(plot_path) if plot_path else None,
        "depths": [int(d) for d in args.depths],
        "num_adp_points": sum(len(v) for v in adp_series.values()),
        "num_stl_points": len(stl_series),
    }
    rg.write_json(task_dir / "generation_depth_sweep_summary.json", summary)
    pd.DataFrame([summary]).to_csv(task_dir / "generation_depth_sweep_summary.csv", index=False)


if __name__ == "__main__":
    main()
