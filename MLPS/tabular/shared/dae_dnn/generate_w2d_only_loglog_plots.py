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


DEFAULT_TASK_CSV_ROOT = "old runs/trial 1/csv/tasks"
DEFAULT_STL_ABLATION_ROOT = "MLPS/tabular/shared/dae_dnn/results/stl_ablation_all_tasks_d3plus_w64plus"
DEFAULT_CURRENT_RUN_ROOT = "MLPS/tabular/shared/dae_dnn/results/goliath_w2d_anomaly_onward_gpu"
DEFAULT_OUTPUT_SUBDIR = "analysis/loss_vs_params_w2d_only_loglog"
REPO_ROOT = Path(__file__).resolve().parents[4]

TASK_DETAILS: Dict[str, Dict[str, Any]] = {
    "classification": {
        "dataset": "Covertype",
        "summary": "Supervised classification learning on standardized Covertype features.",
        "target": "Input: 54 tabular features. Target: 7 forest-cover classes. Training loss: cross-entropy.",
        "in_dim": 54,
        "out_dim": 7,
        "use_bn": True,
    },
    "autoencoding": {
        "dataset": "Covertype",
        "summary": "Reconstruction of standardized Covertype feature vectors.",
        "target": "Input: 54 tabular features. Target: reconstruct the same 54 features. Training loss: MSE.",
        "in_dim": 54,
        "out_dim": 54,
        "use_bn": True,
    },
    "generation": {
        "dataset": "Covertype",
        "summary": "Noise-to-data generation proxy using real Covertype samples as targets.",
        "target": "Input: Gaussian noise vector. Target: real 54-feature Covertype sample. Training loss: MSE.",
        "in_dim": 54,
        "out_dim": 54,
        "use_bn": True,
    },
    "denoising": {
        "dataset": "Covertype",
        "summary": "Tabular denoising autoencoding on standardized Covertype features.",
        "target": "Input: 54-feature Covertype vector with Gaussian noise. Target: clean 54-feature vector. Training loss: MSE.",
        "in_dim": 54,
        "out_dim": 54,
        "use_bn": True,
    },
}

W2D_TASKS = ["classification", "autoencoding", "generation", "denoising"]


@dataclass
class ModelPoint:
    task: str
    family: str
    phase: str
    architecture: List[int]
    best_val: float
    parameter_count: int
    source: str
    label: str

    def to_row(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "family": self.family,
            "phase": self.phase,
            "architecture": rg.format_architecture_for_report(self.architecture),
            "best_val": float(self.best_val),
            "parameter_count": int(self.parameter_count),
            "source": self.source,
            "label": self.label,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate width-to-depth-only ADP vs STL log-log comparison plots.")
    p.add_argument("--task-csv-root", default=DEFAULT_TASK_CSV_ROOT)
    p.add_argument("--stl-ablation-root", default=DEFAULT_STL_ABLATION_ROOT)
    p.add_argument("--current-run-root", default=DEFAULT_CURRENT_RUN_ROOT)
    p.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    p.add_argument("--task", action="append", dest="tasks", default=None, help="Limit generation to specific task(s).")
    return p.parse_args()


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


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


def model_param_count(task_name: str, architecture: Sequence[int]) -> int:
    spec = TASK_DETAILS[task_name]
    model = rg.make_model(
        int(spec["in_dim"]),
        [int(v) for v in architecture],
        int(spec["out_dim"]),
        bool(spec["use_bn"]),
    )
    return int(sum(p.numel() for p in model.parameters()))


def label_for_point(family: str, phase: str, architecture: Sequence[int]) -> str:
    return f"{family} | {phase}\n{rg.format_architecture_for_report(architecture)}"


def collect_w2d_csv_points(task_name: str, csv_root: Path) -> List[ModelPoint]:
    csv_path = csv_root / f"{task_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing archived task CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[
        ((df["family"] == "ADP") & (df["phase"] == "ae_width_to_depth"))
        | ((df["family"] == "STL_paired") & (df["phase"] == "stl_from_ae_width_to_depth"))
    ].copy()

    if df.empty:
        raise ValueError(f"No width-to-depth ADP/STL paired rows found for task {task_name} in {csv_path}")

    points: List[ModelPoint] = []
    for row in df.to_dict(orient="records"):
        architecture = parse_architecture(row["architecture"])
        family = str(row["family"])
        phase = str(row["phase"])
        points.append(
            ModelPoint(
                task=task_name,
                family=family,
                phase=phase,
                architecture=architecture,
                best_val=float(row["best_val"]),
                parameter_count=model_param_count(task_name, architecture),
                source=str(row.get("source", csv_path)),
                label=label_for_point(family, phase, architecture),
            )
        )
    return points


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_ablation_points(task_name: str, stl_ablation_root: Path) -> List[ModelPoint]:
    summary_path = stl_ablation_root / task_name / "ablation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing ablation summary for task {task_name}: {summary_path}")
    summary = load_json(summary_path)
    points: List[ModelPoint] = []
    for entry in summary.get("ablation_stl_runs", []):
        architecture = [int(v) for v in entry["architecture"]]
        phase = str(entry["phase"])
        points.append(
            ModelPoint(
                task=task_name,
                family="STL_ablation",
                phase=phase,
                architecture=architecture,
                best_val=float(entry["best_val"]),
                parameter_count=model_param_count(task_name, architecture),
                source=str(summary_path),
                label=label_for_point("STL_ablation", phase, architecture),
            )
        )
    return points


def write_manifest(output_root: Path, points: Iterable[ModelPoint]) -> None:
    rows = [point.to_row() for point in points]
    pd.DataFrame(rows).to_csv(output_root / "model_manifest.csv", index=False)
    rg.write_json(output_root / "model_manifest.json", {"models": rows})


def plot_task(task_name: str, points: Sequence[ModelPoint], output_root: Path) -> Path:
    fig, ax = plt.subplots(figsize=(24, 18))
    style_map = {
        "ADP": {"marker": "X", "color": "#d62728", "size": 180},
        "STL_paired": {"marker": "s", "color": "#1f77b4", "size": 130},
        "STL_ablation": {"marker": "o", "color": "#2ca02c", "size": 90},
    }

    family_groups: Dict[str, List[ModelPoint]] = {}
    for point in points:
        family_groups.setdefault(point.family, []).append(point)

    for family, family_points in family_groups.items():
        style = style_map[family]
        xs = [point.parameter_count for point in family_points]
        ys = [point.best_val for point in family_points]
        ax.scatter(xs, ys, s=style["size"], marker=style["marker"], color=style["color"], alpha=0.85, label=family)

    for idx, point in enumerate(points):
        dx = 6 if idx % 2 == 0 else -6
        dy = 6 if idx % 3 == 0 else -6
        ax.annotate(
            point.label,
            (point.parameter_count, point.best_val),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7,
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.8", "alpha": 0.85},
        )

    details = TASK_DETAILS[task_name]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Best Validation Loss (log scale)")
    ax.set_title(f"{task_name}: Width-to-Depth ADP vs STL Refit vs STL Ablation", fontsize=18, pad=18)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")

    info_text = (
        f"Task: {task_name}\n"
        f"Dataset: {details['dataset']}\n"
        f"Objective: {details['summary']}\n"
        f"Setup: {details['target']}\n"
        f"Axes: x = total trainable parameters, y = best validation loss.\n"
        f"Filter: ADP rows are restricted to ae_width_to_depth only; paired STL rows are restricted to stl_from_ae_width_to_depth."
    )
    fig.text(
        0.02,
        0.02,
        info_text,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#f8f8f8", "ec": "#cfcfcf", "alpha": 0.95},
    )

    task_dir = output_root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    plot_path = task_dir / f"{task_name}_loss_vs_params_loglog.png"
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    csv_root = resolve_repo_path(args.task_csv_root)
    stl_ablation_root = resolve_repo_path(args.stl_ablation_root)
    current_run_root = resolve_repo_path(args.current_run_root)
    output_root = current_run_root / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or list(W2D_TASKS)
    invalid = [task for task in tasks if task not in TASK_DETAILS]
    if invalid:
        raise SystemExit(f"Unsupported task(s) for width-to-depth archived plotting: {invalid}")

    all_points: List[ModelPoint] = []
    task_plots: Dict[str, str] = {}
    for task_name in tasks:
        task_points = collect_w2d_csv_points(task_name, csv_root) + collect_ablation_points(task_name, stl_ablation_root)
        task_points.sort(key=lambda p: (p.family, p.phase, p.parameter_count, p.best_val))
        all_points.extend(task_points)
        task_plots[task_name] = str(plot_task(task_name, task_points, output_root))

    write_manifest(output_root, all_points)
    rg.write_json(
        output_root / "plot_index.json",
        {
            "tasks": tasks,
            "output_root": str(output_root),
            "task_plots": task_plots,
            "note": "ADP rows are width-to-depth only; paired STL rows are width-to-depth refits only; both axes use log scale.",
        },
    )

    print(f"Generated width-to-depth-only log-log plots in: {output_root}")
    for task_name, plot_path in task_plots.items():
        print(f"{task_name}: {plot_path}")


if __name__ == "__main__":
    main()
