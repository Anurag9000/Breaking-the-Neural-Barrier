from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

import run_goliath as rg


REPO_ROOT = Path(__file__).resolve().parents[4]


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
    p = argparse.ArgumentParser(description="Generate one task's width-to-depth ADP vs STL log-log comparison plot.")
    p.add_argument("--task", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--stl-ablation-root", required=True)
    p.add_argument("--output-subdir", default="analysis/manual_w2d_task_trajectory_loglog")
    p.add_argument("--phase", default="ae_width_to_depth")
    return p.parse_args()


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
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


def is_forced_depth_warmup(architecture: Sequence[int]) -> bool:
    if len(architecture) < 2:
        return False
    prefix = [int(v) for v in architecture[:-1]]
    last = int(architecture[-1])
    return len(set(prefix)) == 1 and last < prefix[0]


def model_param_count(in_dim: int, hidden_widths: Sequence[int], out_dim: int, use_bn: bool) -> int:
    model = rg.make_model(in_dim, [int(v) for v in hidden_widths], out_dim, use_bn)
    return int(sum(p.numel() for p in model.parameters()))


def collect_adp_points(task: str, run_root: Path, phase_name: str) -> List[SeriesPoint]:
    phase_root = run_root / task / phase_name
    if not phase_root.exists():
        raise FileNotFoundError(f"Missing phase root: {phase_root}")
    points: List[SeriesPoint] = []
    for state_path in sorted(phase_root.glob("cand_*/candidate_state.json")):
        candidate_dir = state_path.parent
        candidate_state = load_json(state_path)
        metadata = load_json(candidate_dir / "metadata.json")
        architecture = [int(v) for v in metadata["model"]["hidden_widths"]]
        if is_forced_depth_warmup(architecture):
            continue
        points.append(
            SeriesPoint(
                task=task,
                family="ADP",
                phase=phase_name,
                architecture=architecture,
                best_val=float(candidate_state["best_val"]),
                parameter_count=model_param_count(
                    int(metadata["model"]["in_dim"]),
                    architecture,
                    int(metadata["model"]["out_dim"]),
                    bool(metadata["model"]["use_bn"]),
                ),
                source=str(state_path),
                order_key=int(metadata["candidate_index"]),
                marker_label=candidate_dir.name,
            )
        )
    points.sort(key=lambda p: p.order_key)
    return points


def collect_stl_points(task: str, stl_ablation_root: Path) -> List[SeriesPoint]:
    summary_path = stl_ablation_root / task / "ablation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing STL ablation summary: {summary_path}")
    summary = load_json(summary_path)
    points: List[SeriesPoint] = []
    for entry in summary.get("ablation_stl_runs", []):
        architecture = [int(v) for v in entry["architecture"]]
        candidate_dir = resolve_repo_path(str(entry["checkpoint_best"])).parent
        metadata = load_json(candidate_dir / "metadata.json")
        points.append(
            SeriesPoint(
                task=task,
                family="STL",
                phase=str(entry["phase"]),
                architecture=architecture,
                best_val=float(entry["best_val"]),
                parameter_count=model_param_count(
                    int(metadata["model"]["in_dim"]),
                    architecture,
                    int(metadata["model"]["out_dim"]),
                    bool(metadata["model"]["use_bn"]),
                ),
                source=str(summary_path),
                order_key=model_param_count(
                    int(metadata["model"]["in_dim"]),
                    architecture,
                    int(metadata["model"]["out_dim"]),
                    bool(metadata["model"]["use_bn"]),
                ),
                marker_label=str(entry["phase"]),
            )
        )
    points.sort(key=lambda p: (p.parameter_count, p.best_val, p.phase))
    return points


def plot_task(task: str, task_metadata: Dict[str, Any], adp_points: Sequence[SeriesPoint], stl_points: Sequence[SeriesPoint], output_root: Path) -> Path:
    task_dir = output_root / task
    task_dir.mkdir(parents=True, exist_ok=True)

    rows = [p.to_row() for p in [*adp_points, *stl_points]]
    pd.DataFrame(rows).to_csv(task_dir / f"{task}_trajectory_points.csv", index=False)
    rg.write_json(task_dir / f"{task}_trajectory_points.json", {"points": rows})

    fig, ax = plt.subplots(figsize=(24, 18))

    stl_x = [p.parameter_count for p in stl_points]
    stl_y = [p.best_val for p in stl_points]
    ax.plot(stl_x, stl_y, color="#2ca02c", linewidth=2.5, alpha=0.9, label="STL ablation trend")
    ax.scatter(stl_x, stl_y, color="#2ca02c", marker="o", s=60, alpha=0.9)

    adp_x = [p.parameter_count for p in adp_points]
    adp_y = [p.best_val for p in adp_points]
    ax.plot(adp_x, adp_y, color="#d62728", linewidth=2.2, alpha=0.9, label="ADP width-to-depth trajectory")
    ax.scatter(adp_x, adp_y, color="#d62728", marker="x", s=26, alpha=0.55)

    if adp_points:
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

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Best Validation Loss (log scale)")
    ax.set_title(f"{task}: Width-to-Depth ADP Trajectory vs STL Ablation", fontsize=18, pad=18)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")

    note_lines = [
        f"Task: {task}",
        f"Task type: {task_metadata.get('task_type', 'unknown')}",
        f"Input dim: {task_metadata.get('in_dim', 'unknown')}",
        f"Output dim: {task_metadata.get('out_dim', 'unknown')}",
        "Green line: full STL ablation frontier sorted by parameter count.",
        "Red line: saved ADP width-to-depth candidate trajectory in candidate-index order.",
        "Forced depth warmup candidates are excluded from the red line.",
        "This task was treated as manually completed from the last saved candidate state.",
    ]
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
    run_root = resolve_repo_path(args.run_root)
    stl_ablation_root = resolve_repo_path(args.stl_ablation_root)
    output_root = run_root / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    task_metadata = load_json(run_root / args.task / "task_metadata.json")
    adp_points = collect_adp_points(args.task, run_root, args.phase)
    stl_points = collect_stl_points(args.task, stl_ablation_root)
    plot_path = plot_task(args.task, task_metadata, adp_points, stl_points, output_root)

    manifest_rows = [p.to_row() for p in [*adp_points, *stl_points]]
    pd.DataFrame(manifest_rows).to_csv(output_root / "trajectory_manifest.csv", index=False)
    rg.write_json(output_root / "trajectory_manifest.json", {"points": manifest_rows})
    rg.write_json(
        output_root / "plot_index.json",
        {
            "task": args.task,
            "plot": str(plot_path),
            "manifest_csv": str(output_root / "trajectory_manifest.csv"),
            "manifest_json": str(output_root / "trajectory_manifest.json"),
        },
    )
    print(plot_path)


if __name__ == "__main__":
    main()
