from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from MLPS.tabular.shared.dae_dnn.tasks import build_task
from MLPS.tabular.shared.dae_dnn.runtime_tuning import bootstrap_runtime
from utils.adp_logging import ContinuousLogger

try:  # pragma: no cover - import shim for direct script execution
    import run_goliath as rg
    import run_stl_ablation as stl
except ModuleNotFoundError:  # pragma: no cover - import shim for package-style imports
    from MLPS.tabular.shared.dae_dnn import run_goliath as rg
    from MLPS.tabular.shared.dae_dnn import run_stl_ablation as stl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-task STL planned-parameter CSVs and graphs.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument(
        "--output-root",
        default="MLPS/tabular/shared/dae_dnn/results/stl/ablation/parammatched_decade_v1/analysis/planned_params",
    )
    p.add_argument("--tasks", nargs="+", default=list(stl.DEFAULT_TASKS))
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-width", type=int, default=stl.DEFAULT_MIN_WIDTH)
    p.add_argument("--width-step", type=int, default=stl.DEFAULT_WIDTH_STEP)
    p.add_argument("--width-count-per-depth", type=int, default=stl.DEFAULT_WIDTH_COUNT_PER_DEPTH)
    p.add_argument("--min-depth", type=int, default=stl.DEFAULT_MIN_DEPTH)
    p.add_argument("--max-depth", type=int, default=stl.DEFAULT_MAX_DEPTH)
    p.add_argument("--max-width", type=int, default=stl.DEFAULT_MAX_WIDTH)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument(
        "--param-band",
        nargs=2,
        type=int,
        metavar=("PARAM_EXP_START", "PARAM_EXP_END"),
        default=None,
        help="Optional parameter-count decade band to highlight in the plan report.",
    )
    p.add_argument("--stl-width", type=int, default=128)
    p.add_argument("--stl-depth", type=int, default=2)
    p.add_argument("--use-bn", action="store_true", default=True)
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    p.add_argument(
        "--legacy-architecture-grid",
        action="store_true",
        default=False,
        help="Use the old fixed width x depth sweep instead of parameter-matched depth families.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite the output root if it already exists.")
    return p.parse_args()


def build_cfg(args: argparse.Namespace, tasks: Sequence[str], output_root: Path) -> rg.RunConfig:
    param_band = stl.normalize_param_band(getattr(args, "param_band", None))
    return rg.RunConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        run_root=str(output_root),
        tasks=list(tasks),
        phases=[],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        stl_width=int(args.stl_width),
        stl_depth=int(args.stl_depth),
        alt_start_width=1,
        alt_start_depth=1,
        patience=10,
        width_expansion_patience=10,
        depth_expansion_patience=2,
        delta=1e-4,
        max_epochs=2,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        max_width=int(args.max_width),
        max_depth=int(args.max_depth),
        max_neurons=int(args.max_neurons),
        width_stage_margin_patience=10,
        width_stage_min_improve_pct=1.0,
        use_bn=bool(args.use_bn),
        demo=False,
        metrics_every=0,
        min_width=int(args.min_width),
        width_step=int(args.width_step),
        width_count_per_depth=int(args.width_count_per_depth),
        parameter_matched=not bool(getattr(args, "legacy_architecture_grid", False)),
        parameter_band=param_band,
    )


def log10_safe(value: float) -> float:
    return math.log10(max(1.0, float(value)))


def candidate_parameter_count(task: rg.Task, architecture: Sequence[int], use_bn: bool) -> int:
    return int(stl.parameter_count_for_architecture(task.in_dim, task.out_dim, architecture, use_bn))


def get_depths_for_task(task_name: str, min_depth: int, max_depth: int) -> List[int]:
    task_name = str(task_name).lower()
    allowed = sorted(stl.REMAINING_DEPTHS_BY_TASK.get(task_name, set()))
    return [depth for depth in allowed if int(min_depth) <= depth <= int(max_depth)]


def summarize_task(task_name: str, args: argparse.Namespace, output_root: Path, cfg: rg.RunConfig) -> Dict[str, Any]:
    task = build_task(task_name, cfg.data_dir, 1, cfg.num_workers, cfg.seed, pin_memory=False)
    task_dir = output_root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    param_band = stl.normalize_param_band(getattr(args, "param_band", None))

    target_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    per_depth_counts: List[Dict[str, Any]] = []
    selected_widths_by_depth: Dict[int, List[int]] = {}

    for depth in get_depths_for_task(task_name, int(args.min_depth), int(args.max_depth)):
        min_width = max(1, int(cfg.min_width))
        max_width = max(min_width, int(stl.task_depth_max_width(task_name, depth)))
        min_params = stl._parameter_count_for_width(task, depth, min_width, cfg)
        max_params = stl._parameter_count_for_width(task, depth, max_width, cfg)
        targets = stl.generate_budgeted_parameter_targets(min_params, max_params, int(cfg.width_count_per_depth))
        unique_candidates = stl.parameter_matched_architectures(task, depth, cfg)
        selected_widths = [int(arch[0]) for arch in unique_candidates]
        selected_widths_by_depth[int(depth)] = list(selected_widths)
        seen_widths: set[int] = set()

        for sample_index, target in enumerate(targets, start=1):
            width = int(stl.solve_parameter_matched_width(task, depth, cfg, int(target)))
            param_count = candidate_parameter_count(task, [width] * depth, bool(cfg.use_bn))
            in_band = stl.parameter_target_in_band(int(target), param_band)
            target_rows.append(
                {
                    "task": task_name,
                    "depth": int(depth),
                    "sample_index": int(sample_index),
                    "target_params": int(target),
                    "target_log10_params": round(log10_safe(target), 6),
                    "target_decade": int(math.floor(math.log10(max(1, int(target))))),
                    "solved_width": int(width),
                    "architecture": rg.format_architecture_for_report([width] * depth),
                    "solved_parameter_count": int(param_count),
                    "solved_log10_params": round(log10_safe(param_count), 6),
                    "selected_for_band": bool(in_band),
                    "is_unique_candidate": bool(width not in seen_widths),
                }
            )
            seen_widths.add(width)

        for candidate_index, architecture in enumerate(unique_candidates, start=1):
            width = int(architecture[0])
            param_count = candidate_parameter_count(task, architecture, bool(cfg.use_bn))
            candidate_rows.append(
                {
                    "task": task_name,
                    "depth": int(depth),
                    "candidate_index": int(candidate_index),
                    "width": int(width),
                    "architecture": rg.format_architecture_for_report(architecture),
                    "parameter_count": int(param_count),
                    "log10_parameter_count": round(log10_safe(param_count), 6),
                    "selected_for_run": True,
                }
            )

        decade_counts = Counter(int(row["target_decade"]) for row in target_rows if int(row["depth"]) == int(depth))
        per_depth_counts.append(
            {
                "task": task_name,
                "depth": int(depth),
                "target_count": len(targets),
                "selected_candidate_count": len(unique_candidates),
                "unique_solved_width_count": len(selected_widths),
                "decade_counts": dict(sorted(decade_counts.items())),
            }
        )

    rg.write_csv(
        task_dir / "planned_target_samples.csv",
        target_rows,
        fieldnames=[
            "task",
            "depth",
            "sample_index",
            "target_params",
            "target_log10_params",
            "target_decade",
            "solved_width",
            "architecture",
            "solved_parameter_count",
            "solved_log10_params",
            "selected_for_band",
            "is_unique_candidate",
        ],
    )
    rg.write_csv(
        task_dir / "planned_candidate_families.csv",
        candidate_rows,
        fieldnames=[
            "task",
            "depth",
            "candidate_index",
            "width",
            "architecture",
            "parameter_count",
            "log10_parameter_count",
            "selected_for_run",
        ],
    )
    rg.write_json(
        task_dir / "planned_summary.json",
        {
            "task": task_name,
            "param_band": list(param_band) if param_band is not None else None,
            "depths": get_depths_for_task(task_name, int(args.min_depth), int(args.max_depth)),
            "selected_widths_by_depth": selected_widths_by_depth,
            "per_depth_counts": per_depth_counts,
            "target_row_count": len(target_rows),
            "candidate_row_count": len(candidate_rows),
            "candidate_count_by_depth": {int(item["depth"]): int(item["selected_candidate_count"]) for item in per_depth_counts},
        },
    )
    plot_path = render_task_plot(task_dir, task_name, target_rows, candidate_rows, param_band)
    return {
        "task": task_name,
        "target_row_count": len(target_rows),
        "candidate_row_count": len(candidate_rows),
        "plot_path": str(plot_path),
    }


def render_task_plot(
    task_dir: Path,
    task_name: str,
    target_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    param_band: Optional[Tuple[int, int]],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2.2, 1.0]})
    ax_scatter, ax_hist = axes

    depths = sorted({int(row["depth"]) for row in target_rows})
    cmap = plt.get_cmap("tab10")
    for idx, depth in enumerate(depths):
        depth_targets = [row for row in target_rows if int(row["depth"]) == depth]
        if not depth_targets:
            continue
        xs = [depth for _ in depth_targets]
        ys = [float(row["target_log10_params"]) for row in depth_targets]
        ax_scatter.scatter(xs, ys, s=18, alpha=0.2, color="0.5", edgecolors="none")

        depth_candidates = [row for row in candidate_rows if int(row["depth"]) == depth]
        if depth_candidates:
            cand_x = [depth for _ in depth_candidates]
            cand_y = [float(row["log10_parameter_count"]) for row in depth_candidates]
            ax_scatter.scatter(
                cand_x,
                cand_y,
                s=42,
                alpha=0.9,
                color=cmap(idx % 10),
                label=f"depth {depth}",
                edgecolors="black",
                linewidths=0.3,
            )
            for row in depth_candidates:
                ax_scatter.annotate(
                    str(int(row["width"])),
                    (depth + 0.04, float(row["log10_parameter_count"])),
                    fontsize=7,
                    alpha=0.75,
                )

    if target_rows:
        min_log = min(float(row["target_log10_params"]) for row in target_rows)
        max_log = max(float(row["target_log10_params"]) for row in target_rows)
        lower = int(math.floor(min_log))
        upper = int(math.ceil(max_log))
        for decade in range(lower, upper + 1):
            ax_scatter.axhline(decade, color="0.8", linewidth=0.7, linestyle="--", alpha=0.6)
            ax_hist.axhline(decade, color="0.9", linewidth=0.7, linestyle="--", alpha=0.35)

    ax_scatter.set_title(f"{task_name}: planned candidates by depth")
    ax_scatter.set_xlabel("Depth")
    ax_scatter.set_ylabel("log10(parameter count)")
    ax_scatter.set_xticks(depths)
    ax_scatter.grid(True, alpha=0.2)
    if candidate_rows:
        ax_scatter.legend(loc="best", fontsize=8, frameon=False)

    target_decades = [int(row["target_decade"]) for row in target_rows]
    candidate_decades = [int(math.floor(float(row["log10_parameter_count"]))) for row in candidate_rows]
    decade_bins = sorted(set(target_decades + candidate_decades))
    if decade_bins:
        target_counts = [target_decades.count(decade) for decade in decade_bins]
        candidate_counts = [candidate_decades.count(decade) for decade in decade_bins]
        x = list(range(len(decade_bins)))
        width = 0.38
        ax_hist.bar([v - width / 2 for v in x], target_counts, width=width, color="0.7", label="sample targets")
        ax_hist.bar([v + width / 2 for v in x], candidate_counts, width=width, color="tab:blue", label="unique candidates")
        ax_hist.set_xticks(x)
        ax_hist.set_xticklabels([f"10^{d}" for d in decade_bins], rotation=45, ha="right")
    ax_hist.set_title("Counts by decade")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, axis="y", alpha=0.2)
    ax_hist.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(
        f"{task_name}: planned parameter distribution"
        + (f" band {param_band[0]}..{param_band[1]}" if param_band is not None else ""),
        fontsize=15,
    )
    fig.tight_layout()

    plot_path = task_dir / "planned_params_decade_distribution.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    rg.write_json(
        task_dir / "planned_plot.json",
        {
            "task": task_name,
            "plot_path": str(plot_path),
            "note": "Gray points are all raw decade samples; colored points are the unique candidate families actually run.",
        },
    )
    return plot_path


def main() -> None:
    bootstrap_runtime("generate_stl_planned_params_report")

    args = parse_args()
    tasks = [str(task).lower() for task in args.tasks]
    output_root = Path(args.output_root)
    if output_root.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    band = stl.normalize_param_band(getattr(args, "param_band", None))
    cfg = build_cfg(args, tasks, output_root)

    logger = ContinuousLogger(output_root, "stl_planned_params", "stl_planned_params")
    logger.log_console(f"Output root: {output_root}")
    logger.log_console(f"Tasks: {tasks}")
    logger.log_console(f"Parameter band: {list(band) if band is not None else 'all'}")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    reports: List[Dict[str, Any]] = []
    try:
        for task_name in tasks:
            logger.log_console(f"[TASK] start {task_name}")
            report = summarize_task(task_name, args, output_root, cfg)
            reports.append(report)

        # Re-read the per-task CSVs for the combined root-level plan.
        all_target_rows: List[Dict[str, Any]] = []
        all_candidate_rows: List[Dict[str, Any]] = []
        for task_name in tasks:
            task_dir = output_root / task_name
            target_path = task_dir / "planned_target_samples.csv"
            candidate_path = task_dir / "planned_candidate_families.csv"
            if target_path.exists():
                with target_path.open("r", encoding="utf-8", newline="") as f:
                    all_target_rows.extend(list(csv.DictReader(f)))
            if candidate_path.exists():
                with candidate_path.open("r", encoding="utf-8", newline="") as f:
                    all_candidate_rows.extend(list(csv.DictReader(f)))

        rg.write_csv(
            output_root / "planned_params_by_task_depth_width.csv",
            all_candidate_rows,
            fieldnames=["task", "depth", "candidate_index", "width", "architecture", "parameter_count", "log10_parameter_count", "selected_for_run"],
        )
        rg.write_csv(
            output_root / "planned_target_samples_by_task_depth_width.csv",
            all_target_rows,
            fieldnames=[
                "task",
                "depth",
                "sample_index",
                "target_params",
                "target_log10_params",
                "target_decade",
                "solved_width",
                "architecture",
                "solved_parameter_count",
                "solved_log10_params",
                "selected_for_band",
                "is_unique_candidate",
            ],
        )
        rg.write_json(
            output_root / "planned_params_summary.json",
            {
                "tasks": tasks,
                "param_band": list(band) if band is not None else None,
                "reports": reports,
                "candidate_row_count": len(all_candidate_rows),
                "target_row_count": len(all_target_rows),
            },
        )
    finally:
        logger.close()


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
