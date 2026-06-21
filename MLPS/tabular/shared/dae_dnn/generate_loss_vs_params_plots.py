from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch

import run_goliath as rg


DEFAULT_ADP_RUN_ROOT = "MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current"
DEFAULT_STL_ABLATION_ROOT = "MLPS/tabular/shared/dae_dnn/results/stl_ablation_all_tasks_d3plus_w64plus"
DEFAULT_OUTPUT_ROOT = "MLPS/tabular/shared/dae_dnn/results/analysis/loss_vs_params_completed_both"


TASK_DETAILS: Dict[str, Dict[str, str]] = {
    "classification": {
        "dataset": "Covertype",
        "summary": "Classification-style classification learning on standardized Covertype features.",
        "target": "Input: 54 tabular features. Target: 7 forest-cover classes. Validation loss: cross-entropy.",
    },
    "autoencoding": {
        "dataset": "Covertype",
        "summary": "Reconstruction task on standardized Covertype tabular vectors.",
        "target": "Input: 54 tabular features. Target: reconstruct the same 54 features. Validation loss: MSE.",
    },
    "generation": {
        "dataset": "Covertype",
        "summary": "Noise-to-data generation proxy using real Covertype samples as targets.",
        "target": "Input: Gaussian noise vector. Target: real 54-feature Covertype sample. Validation loss: MSE.",
    },
    "denoising": {
        "dataset": "Covertype",
        "summary": "Tabular denoising autoencoding on standardized Covertype features.",
        "target": "Input: 54-feature Covertype vector corrupted with Gaussian noise. Target: clean 54-feature vector. Validation loss: MSE.",
    },
}


@dataclass
class ModelPoint:
    task: str
    family: str
    phase: str
    architecture: List[int]
    best_val: float
    checkpoint_best: str
    parameter_count: int
    label: str

    def to_row(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "family": self.family,
            "phase": self.phase,
            "architecture": rg.format_architecture_for_report(self.architecture),
            "best_val": float(self.best_val),
            "parameter_count": int(self.parameter_count),
            "checkpoint_best": self.checkpoint_best,
            "label": self.label,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate loss-vs-parameter plots for completed ADP/STL tabular tasks.")
    p.add_argument("--adp-run-root", default=DEFAULT_ADP_RUN_ROOT)
    p.add_argument("--stl-ablation-root", default=DEFAULT_STL_ABLATION_ROOT)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--task", default=None, help="Generate only for a specific task.")
    p.add_argument("--allow-partial", action="store_true", help="Allow generating a task from completed ADP phases even if the task is not fully completed.")
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def completed_adp_tasks(run_root: Path) -> List[str]:
    tasks: List[str] = []
    for path in sorted(run_root.glob("*/task_state.json")):
        data = rg.load_json_if_exists(path) or {}
        if data.get("completed") is True:
            tasks.append(path.parent.name)
    return tasks


def completed_stl_ablation_tasks(run_root: Path) -> List[str]:
    tasks: List[str] = []
    for path in sorted(run_root.glob("*/ablation_summary.json")):
        data = rg.load_json_if_exists(path) or {}
        if isinstance(data, dict) and data.get("ablation_stl_runs"):
            tasks.append(path.parent.name)
    return tasks


def parameter_count_from_metadata(candidate_dir: Path) -> int:
    metadata = load_json(candidate_dir / "metadata.json")
    model_cfg = metadata["model"]
    model = rg.make_model(
        int(model_cfg["in_dim"]),
        [int(v) for v in model_cfg["hidden_widths"]],
        int(model_cfg["out_dim"]),
        bool(model_cfg["use_bn"]),
    )
    return int(sum(p.numel() for p in model.parameters()))


def label_for_point(family: str, phase: str, architecture: Sequence[int]) -> str:
    return f"{family} | {phase}\n{rg.format_architecture_for_report(architecture)}"


def collect_adp_and_paired_stl(task_root: Path) -> List[ModelPoint]:
    summary = load_json(task_root / "task_summary.json")
    task_name = str(summary["task"])
    points: List[ModelPoint] = []

    for entry in summary.get("adp_runs", []):
        phase = str(entry["phase"])
        architecture = [int(v) for v in entry["architecture"]]
        candidate_dir = task_root / phase / str(entry["best_candidate_dir"])
        points.append(
            ModelPoint(
                task=task_name,
                family="ADP",
                phase=phase,
                architecture=architecture,
                best_val=float(entry["best_val"]),
                checkpoint_best=str(candidate_dir / "checkpoint_best.pt"),
                parameter_count=parameter_count_from_metadata(candidate_dir),
                label=label_for_point("ADP", phase, architecture),
            )
        )

    for entry in summary.get("paired_stl_runs", []):
        phase = str(entry["phase"])
        architecture = [int(v) for v in entry["architecture"]]
        candidate_dir = task_root / phase / str(entry["candidate_dir"])
        points.append(
            ModelPoint(
                task=task_name,
                family="STL_paired",
                phase=phase,
                architecture=architecture,
                best_val=float(entry["best_val"]),
                checkpoint_best=str(candidate_dir / "checkpoint_best.pt"),
                parameter_count=parameter_count_from_metadata(candidate_dir),
                label=label_for_point("STL_paired", phase, architecture),
            )
        )

    return points


def collect_partial_adp_and_paired_stl(task_root: Path) -> List[ModelPoint]:
    task_state = load_json(task_root / "task_state.json")
    completed = set(task_state.get("completed_phases", []))
    task_name = str(task_state["task"])
    points: List[ModelPoint] = []

    for phase in sorted(p.name for p in task_root.iterdir() if p.is_dir() and p.name.startswith("ae_")):
        if phase not in completed:
            continue
        phase_summary_path = task_root / phase / "phase_summary.json"
        if not phase_summary_path.exists():
            continue
        entry = load_json(phase_summary_path)
        architecture = [int(v) for v in entry["architecture"]]
        candidate_dir = task_root / phase / str(entry["best_candidate_dir"])
        points.append(
            ModelPoint(
                task=task_name,
                family="ADP",
                phase=phase,
                architecture=architecture,
                best_val=float(entry["best_val"]),
                checkpoint_best=str(candidate_dir / "checkpoint_best.pt"),
                parameter_count=parameter_count_from_metadata(candidate_dir),
                label=label_for_point("ADP", phase, architecture),
            )
        )

    for phase in sorted(p.name for p in task_root.iterdir() if p.is_dir() and p.name.startswith("stl_from_")):
        if phase not in completed:
            continue
        phase_summary_path = task_root / phase / "phase_summary.json"
        if not phase_summary_path.exists():
            continue
        entry = load_json(phase_summary_path)
        architecture = [int(v) for v in entry["architecture"]]
        candidate_dir = task_root / phase / str(entry["candidate_dir"])
        points.append(
            ModelPoint(
                task=task_name,
                family="STL_paired",
                phase=phase,
                architecture=architecture,
                best_val=float(entry["best_val"]),
                checkpoint_best=str(candidate_dir / "checkpoint_best.pt"),
                parameter_count=parameter_count_from_metadata(candidate_dir),
                label=label_for_point("STL_paired", phase, architecture),
            )
        )

    return points


def collect_ablation_stl(task_root: Path) -> List[ModelPoint]:
    summary = load_json(task_root / "ablation_summary.json")
    task_name = str(summary["task"])
    points: List[ModelPoint] = []
    for entry in summary.get("ablation_stl_runs", []):
        architecture = [int(v) for v in entry["architecture"]]
        checkpoint_best = str(entry["checkpoint_best"])
        candidate_dir = Path(checkpoint_best).parent
        phase = str(entry["phase"])
        points.append(
            ModelPoint(
                task=task_name,
                family="STL_ablation",
                phase=phase,
                architecture=architecture,
                best_val=float(entry["best_val"]),
                checkpoint_best=checkpoint_best,
                parameter_count=parameter_count_from_metadata(candidate_dir),
                label=label_for_point("STL_ablation", phase, architecture),
            )
        )
    return points


def write_manifest(output_root: Path, points: Iterable[ModelPoint]) -> Path:
    rows = [point.to_row() for point in points]
    manifest_path = output_root / "model_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    rg.write_json(output_root / "model_manifest.json", {"models": rows})
    return manifest_path


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

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Best Validation Loss")
    details = TASK_DETAILS.get(
        task_name,
        {
            "dataset": "Unknown dataset",
            "summary": "Task summary unavailable.",
            "target": "See saved task metadata for exact input/target semantics.",
        },
    )
    ax.set_title(f"{task_name}: Loss vs Parameters for ADP and STL Models", fontsize=18, pad=18)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")

    info_text = (
        f"Task: {task_name}\n"
        f"Dataset: {details['dataset']}\n"
        f"Objective: {details['summary']}\n"
        f"Setup: {details['target']}\n"
        f"Axes: x = total trainable parameters, y = best validation loss.\n"
        f"Labels: family | phase | architecture. ADP phases are one of ae_alt_depth, ae_alt_width, "
        f"ae_width_to_depth, ae_depth_to_width."
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
    plot_path = task_dir / f"{task_name}_loss_vs_params.png"
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    adp_root = Path(args.adp_run_root)
    stl_root = Path(args.stl_ablation_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    both_completed = sorted(set(completed_adp_tasks(adp_root)).intersection(completed_stl_ablation_tasks(stl_root)))

    if args.task:
        both_completed = [args.task]

    if not both_completed:
        raise SystemExit("No tasks are completed in both the ADP run root and the STL ablation run root.")

    all_points: List[ModelPoint] = []
    task_plot_paths: Dict[str, str] = {}

    for task_name in both_completed:
        adp_task_root = adp_root / task_name
        if (adp_task_root / "task_state.json").exists():
            task_state = load_json(adp_task_root / "task_state.json")
            adp_done = bool(task_state.get("completed"))
        else:
            adp_done = False

        if adp_done:
            task_points = collect_adp_and_paired_stl(adp_task_root)
        elif args.allow_partial:
            task_points = collect_partial_adp_and_paired_stl(adp_task_root)
        else:
            raise SystemExit(f"Task {task_name} is not fully completed in ADP run root; rerun with --allow-partial to include only completed phases.")

        task_points += collect_ablation_stl(stl_root / task_name)
        task_points.sort(key=lambda p: (p.family, p.phase, p.parameter_count, p.best_val))
        all_points.extend(task_points)
        task_plot_paths[task_name] = str(plot_task(task_name, task_points, output_root))

    manifest_path = write_manifest(output_root, all_points)

    combined_path = output_root / "all_completed_tasks_loss_vs_params.png"
    if combined_path.exists():
        combined_path.unlink()

    rg.write_json(
        output_root / "plot_index.json",
        {
            "tasks_completed_by_both": both_completed,
            "manifest_csv": str(manifest_path),
            "task_plots": task_plot_paths,
        },
    )

    print(f"Generated plots in: {output_root}")
    for task_name, plot_path in task_plot_paths.items():
        print(f"{task_name}: {plot_path}")
    print(f"manifest: {manifest_path}")


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
