from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

import run_goliath as rg


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CURRENT_RUN_ROOT = "MLPS/tabular/shared/dae_dnn/results/goliath_w2d_anomaly_onward_gpu"
DEFAULT_RECOVERED_ROOT = "MLPS/tabular/shared/dae_dnn/results/goliath_w2d_anomaly_onward_gpu/analysis/recovered_trial1_w2d_history"
DEFAULT_STL_ABLATION_ROOT = "MLPS/tabular/shared/dae_dnn/results/stl_ablation_all_tasks_d3plus_w64plus"
DEFAULT_OUTPUT_SUBDIR = "analysis/loss_vs_params_w2d_trajectory_loglog"

TASK_DETAILS: Dict[str, Dict[str, str]] = {
    "representation": {
        "dataset": "Covertype",
        "summary": "Supervised representation learning on standardized Covertype features.",
        "target": "Input: 54 tabular features. Target: 7 forest-cover classes. Training loss: cross-entropy.",
    },
    "autoencoding": {
        "dataset": "Covertype",
        "summary": "Reconstruction of standardized Covertype feature vectors.",
        "target": "Input: 54 tabular features. Target: reconstruct the same 54 features. Training loss: MSE.",
    },
    "generation": {
        "dataset": "Covertype",
        "summary": "Noise-to-data generation proxy using real Covertype samples as targets.",
        "target": "Input: Gaussian noise vector. Target: real 54-feature Covertype sample. Training loss: MSE.",
    },
    "denoising": {
        "dataset": "Covertype",
        "summary": "Tabular denoising autoencoding on standardized Covertype features.",
        "target": "Input: 54-feature Covertype vector with Gaussian noise. Target: clean 54-feature vector. Training loss: MSE.",
    },
    "anomaly": {
        "dataset": "Covertype",
        "summary": "One-class reconstruction on normal Covertype samples; anomaly score is reconstruction error.",
        "target": "Input: 54 tabular features from the normal class during training. Target: reconstruct the same 54 features. Training loss: MSE.",
    },
}

OLD_COMPLETED_TASKS = ["representation", "autoencoding", "generation", "denoising"]
CURRENT_TASKS = OLD_COMPLETED_TASKS + ["anomaly"]


@dataclass
class SeriesPoint:
    task: str
    family: str
    phase: str
    architecture: List[int]
    best_val: float
    parameter_count: int
    source: str
    order_key: int
    marker_label: str

    def to_row(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "family": self.family,
            "phase": self.phase,
            "architecture": rg.format_architecture_for_report(self.architecture),
            "best_val": float(self.best_val),
            "parameter_count": int(self.parameter_count),
            "source": self.source,
            "order_key": int(self.order_key),
            "marker_label": self.marker_label,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate width-to-depth ADP trajectory vs STL trend log-log plots.")
    p.add_argument("--current-run-root", default=DEFAULT_CURRENT_RUN_ROOT)
    p.add_argument("--recovered-root", default=DEFAULT_RECOVERED_ROOT)
    p.add_argument("--stl-ablation-root", default=DEFAULT_STL_ABLATION_ROOT)
    p.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    return p.parse_args()


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_recovered_summary(recovered_root: Path) -> pd.DataFrame:
    csv_path = recovered_root / "recovered_candidate_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing recovered summary CSV: {csv_path}")
    return pd.read_csv(csv_path)


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


def candidate_index_from_name(name: str) -> int:
    return int(name.split("_", 2)[1])


def collect_old_adp_points(task: str, recovered_root: Path) -> List[SeriesPoint]:
    df = read_recovered_summary(recovered_root)
    df = df[(df["task"] == task) & (df["phase"] == "ae_width_to_depth")].copy()
    if df.empty:
        return []
    points: List[SeriesPoint] = []
    for row in df.to_dict(orient="records"):
        architecture = parse_architecture(row["architecture"])
        candidate_name = str(row["candidate_name"])
        points.append(
            SeriesPoint(
                task=task,
                family="ADP",
                phase="ae_width_to_depth",
                architecture=architecture,
                best_val=float(row["best_val"]),
                parameter_count=int(row["parameter_count"]),
                source=str(row["candidate_state_json"]),
                order_key=candidate_index_from_name(candidate_name),
                marker_label=candidate_name,
            )
        )
    points.sort(key=lambda p: p.order_key)
    return points


def collect_old_paired_stl_points(task: str, recovered_root: Path) -> List[SeriesPoint]:
    df = read_recovered_summary(recovered_root)
    df = df[(df["task"] == task) & (df["phase"] == "stl_from_ae_width_to_depth")].copy()
    points: List[SeriesPoint] = []
    for row in df.to_dict(orient="records"):
        architecture = parse_architecture(row["architecture"])
        candidate_name = str(row["candidate_name"])
        points.append(
            SeriesPoint(
                task=task,
                family="STL_paired",
                phase="stl_from_ae_width_to_depth",
                architecture=architecture,
                best_val=float(row["best_val"]),
                parameter_count=int(row["parameter_count"]),
                source=str(row["candidate_state_json"]),
                order_key=int(row["parameter_count"]),
                marker_label=candidate_name,
            )
        )
    return points


def collect_current_anomaly_adp_points(current_run_root: Path) -> List[SeriesPoint]:
    phase_root = current_run_root / "anomaly" / "ae_width_to_depth"
    if not phase_root.exists():
        raise FileNotFoundError(f"Missing current anomaly phase root: {phase_root}")
    points: List[SeriesPoint] = []
    for state_path in sorted(phase_root.glob("cand_*/candidate_state.json")):
        candidate_dir = state_path.parent
        candidate_state = load_json(state_path)
        metadata = load_json(candidate_dir / "metadata.json")
        architecture = [int(v) for v in metadata["model"]["hidden_widths"]]
        parameter_count = int(sum(p.numel() for p in rg.make_model(
            int(metadata["model"]["in_dim"]),
            architecture,
            int(metadata["model"]["out_dim"]),
            bool(metadata["model"]["use_bn"]),
        ).parameters()))
        points.append(
            SeriesPoint(
                task="anomaly",
                family="ADP",
                phase="ae_width_to_depth",
                architecture=architecture,
                best_val=float(candidate_state["best_val"]),
                parameter_count=parameter_count,
                source=str(state_path),
                order_key=int(metadata["candidate_index"]),
                marker_label=candidate_dir.name,
            )
        )
    points.sort(key=lambda p: p.order_key)
    return points


def collect_ablation_stl_points(task: str, stl_ablation_root: Path) -> List[SeriesPoint]:
    summary_path = stl_ablation_root / task / "ablation_summary.json"
    if not summary_path.exists():
        return []
    summary = load_json(summary_path)
    points: List[SeriesPoint] = []
    for entry in summary.get("ablation_stl_runs", []):
        architecture = [int(v) for v in entry["architecture"]]
        candidate_dir = resolve_repo_path(str(entry["checkpoint_best"])).parent
        metadata = load_json(candidate_dir / "metadata.json")
        parameter_count = int(sum(p.numel() for p in rg.make_model(
            int(metadata["model"]["in_dim"]),
            architecture,
            int(metadata["model"]["out_dim"]),
            bool(metadata["model"]["use_bn"]),
        ).parameters()))
        points.append(
            SeriesPoint(
                task=task,
                family="STL_ablation",
                phase=str(entry["phase"]),
                architecture=architecture,
                best_val=float(entry["best_val"]),
                parameter_count=parameter_count,
                source=str(summary_path),
                order_key=parameter_count,
                marker_label=str(entry["phase"]),
            )
        )
    points.sort(key=lambda p: (p.parameter_count, p.best_val))
    return points


def write_task_csv(task_dir: Path, task: str, adp_points: Sequence[SeriesPoint], stl_points: Sequence[SeriesPoint]) -> None:
    rows = [point.to_row() for point in [*adp_points, *stl_points]]
    pd.DataFrame(rows).to_csv(task_dir / f"{task}_trajectory_points.csv", index=False)
    rg.write_json(task_dir / f"{task}_trajectory_points.json", {"points": rows})


def plot_task(
    task: str,
    adp_points: Sequence[SeriesPoint],
    stl_points: Sequence[SeriesPoint],
    output_root: Path,
    partial_note: str | None = None,
) -> Path:
    task_dir = output_root / task
    task_dir.mkdir(parents=True, exist_ok=True)
    write_task_csv(task_dir, task, adp_points, stl_points)

    fig, ax = plt.subplots(figsize=(24, 18))

    if stl_points:
        stl_x = [p.parameter_count for p in stl_points]
        stl_y = [p.best_val for p in stl_points]
        ax.plot(stl_x, stl_y, color="#2ca02c", linewidth=2.5, alpha=0.9, label="STL trend")
        ax.scatter(stl_x, stl_y, color="#2ca02c", marker="o", s=60, alpha=0.9)

    if adp_points:
        adp_x = [p.parameter_count for p in adp_points]
        adp_y = [p.best_val for p in adp_points]
        ax.plot(adp_x, adp_y, color="#d62728", linewidth=2.2, alpha=0.9, label="ADP width-to-depth trajectory")
        ax.scatter(adp_x, adp_y, color="#d62728", marker="x", s=26, alpha=0.55)
        best_point = min(adp_points, key=lambda p: p.best_val)
        last_point = max(adp_points, key=lambda p: p.order_key)
        ax.annotate(
            f"best {rg.format_architecture_for_report(best_point.architecture)}\n{best_point.best_val:.6g}",
            (best_point.parameter_count, best_point.best_val),
            textcoords="offset points",
            xytext=(8, -10),
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.8", "alpha": 0.9},
        )
        if last_point.order_key != best_point.order_key:
            ax.annotate(
                f"last {rg.format_architecture_for_report(last_point.architecture)}\n{last_point.best_val:.6g}",
                (last_point.parameter_count, last_point.best_val),
                textcoords="offset points",
                xytext=(8, 10),
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.8", "alpha": 0.9},
            )

    details = TASK_DETAILS[task]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Best Validation Loss (log scale)")
    ax.set_title(f"{task}: Width-to-Depth ADP Search Trajectory vs STL Trend", fontsize=18, pad=18)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")

    note_lines = [
        f"Task: {task}",
        f"Dataset: {details['dataset']}",
        f"Objective: {details['summary']}",
        f"Setup: {details['target']}",
        "Green line: STL ablations plus paired STL refit(s), sorted by parameter count.",
        "Red line: ADP width-to-depth candidate trajectory, in candidate-index order.",
    ]
    if partial_note:
        note_lines.append(partial_note)
    fig.text(
        0.02,
        0.02,
        "\n".join(note_lines),
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#f8f8f8", "ec": "#cfcfcf", "alpha": 0.95},
    )

    plot_path = task_dir / f"{task}_trajectory_loss_vs_params_loglog.png"
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    current_run_root = resolve_repo_path(args.current_run_root)
    recovered_root = resolve_repo_path(args.recovered_root)
    stl_ablation_root = resolve_repo_path(args.stl_ablation_root)
    output_root = current_run_root / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    task_plots: Dict[str, str] = {}

    for task in OLD_COMPLETED_TASKS:
        adp_points = collect_old_adp_points(task, recovered_root)
        stl_points = collect_ablation_stl_points(task, stl_ablation_root) + collect_old_paired_stl_points(task, recovered_root)
        stl_points.sort(key=lambda p: (p.parameter_count, p.best_val, p.phase))
        plot_path = plot_task(task, adp_points, stl_points, output_root)
        task_plots[task] = str(plot_path)
        all_rows.extend(point.to_row() for point in [*adp_points, *stl_points])

    anomaly_adp = collect_current_anomaly_adp_points(current_run_root)
    anomaly_stl = collect_ablation_stl_points("anomaly", stl_ablation_root)
    anomaly_partial_note = "Anomaly ADP is a live partial run; the red trajectory includes all saved candidates through the latest checkpointed state."
    plot_path = plot_task("anomaly", anomaly_adp, anomaly_stl, output_root, partial_note=anomaly_partial_note)
    task_plots["anomaly"] = str(plot_path)
    all_rows.extend(point.to_row() for point in [*anomaly_adp, *anomaly_stl])

    pd.DataFrame(all_rows).to_csv(output_root / "trajectory_manifest.csv", index=False)
    rg.write_json(output_root / "trajectory_manifest.json", {"points": all_rows})
    rg.write_json(
        output_root / "plot_index.json",
        {
            "tasks": CURRENT_TASKS,
            "task_plots": task_plots,
            "manifest_csv": str(output_root / "trajectory_manifest.csv"),
            "manifest_json": str(output_root / "trajectory_manifest.json"),
        },
    )

    print(f"Generated trajectory plots in: {output_root}")
    for task, plot_path in task_plots.items():
        print(f"{task}: {plot_path}")


if __name__ == "__main__":
    main()
