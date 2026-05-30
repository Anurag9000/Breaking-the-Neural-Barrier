from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

from DAE.DNN.tasks import build_task
from utils.adp_logging import ContinuousLogger

import run_goliath as rg


DEFAULT_TASKS = [
    "representation",
    "autoencoding",
    "generation",
    "denoising",
    "anomaly",
    "simulation",
    "prediction",
]

DEFAULT_MIN_DEPTH = 1
DEFAULT_MAX_DEPTH = 10
DEFAULT_MIN_WIDTH = 16
DEFAULT_MAX_WIDTH = 1024
DEFAULT_WIDTH_STEP = 16
DEFAULT_REPEAT_COUNT = 5


def parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_architectures(values: Sequence[str]) -> List[List[int]]:
    architectures: List[List[int]] = []
    for item in values:
        arch = parse_csv_ints(item)
        if arch:
            architectures.append([max(1, int(v)) for v in arch])
    return architectures


def dedupe_architectures(architectures: Iterable[Sequence[int]]) -> List[List[int]]:
    seen = set()
    out: List[List[int]] = []
    for arch in architectures:
        key = tuple(int(v) for v in arch)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def build_architectures(args) -> List[List[int]]:
    if args.architecture:
        return dedupe_architectures(parse_architectures(args.architecture))
    if args.widths and args.depths:
        widths = parse_csv_ints(args.widths)
        depths = parse_csv_ints(args.depths)
        return dedupe_architectures([[int(width)] * int(depth) for depth in depths for width in widths])
    min_depth = max(1, int(args.min_depth))
    max_depth = max(min_depth, int(args.max_depth))
    min_width = max(1, int(args.min_width))
    max_width = max(min_width, int(args.max_width))
    width_step = max(1, int(args.width_step))
    widths = list(range(min_width, max_width + 1, width_step))
    depths = list(range(min_depth, max_depth + 1))
    return dedupe_architectures([[int(width)] * int(depth) for depth in depths for width in widths])


def phase_name_for_architecture(architecture: Sequence[int], repeat_index: int) -> str:
    depth = len(architecture)
    width = max(int(v) for v in architecture)
    return f"stl_ablation_r{repeat_index:02d}_d{depth:02d}_w{width:04d}_{'_'.join(str(int(v)) for v in architecture)}"


def load_source_task_summary(source_run_root: Path, task_name: str) -> Dict[str, Any]:
    path = source_run_root / task_name / "task_summary.json"
    return rg.load_json_if_exists(path) or {}


def comparison_row(
    *,
    task_name: str,
    repeat_index: int,
    ablation_phase: str,
    ablation_architecture: Sequence[int],
    ablation_parameter_count: int,
    ablation_best_val: float,
    ref_phase: str,
    ref_kind: str,
    ref_architecture: Optional[Sequence[int]],
    ref_best_val: float,
) -> Dict[str, Any]:
    winner = "ablation_stl" if float(ablation_best_val) <= float(ref_best_val) else ref_kind
    return {
        "task": task_name,
        "repeat": int(repeat_index),
        "ablation_phase": ablation_phase,
        "ablation_architecture": rg.format_architecture_for_report(ablation_architecture),
        "ablation_parameter_count": int(ablation_parameter_count),
        "ablation_best_val": float(ablation_best_val),
        "reference_kind": ref_kind,
        "reference_phase": ref_phase,
        "reference_architecture": rg.format_architecture_for_report(ref_architecture),
        "reference_best_val": float(ref_best_val),
        "winner": winner,
        "winner_value": min(float(ablation_best_val), float(ref_best_val)),
    }


def parameter_count_for_summary(task: rg.Task, task_root: Path, summary: Dict[str, Any]) -> int:
    phase = str(summary["phase"])
    candidate_dir = task_root / phase / str(summary["candidate_dir"])
    metadata = rg.load_json_if_exists(candidate_dir / "metadata.json") or {}
    model_cfg = metadata.get("model") or {}
    model = rg.make_model(
        int(model_cfg.get("in_dim", task.in_dim)),
        [int(v) for v in model_cfg.get("hidden_widths", summary.get("architecture", []))],
        int(model_cfg.get("out_dim", task.out_dim)),
        bool(model_cfg.get("use_bn", True)),
    )
    return int(rg.count_model_parameters(model))


def make_cfg(args, tasks: List[str], run_root: Path) -> rg.RunConfig:
    return rg.RunConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        run_root=str(run_root),
        tasks=tasks,
        phases=[],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        stl_width=int(args.stl_width),
        stl_depth=int(args.stl_depth),
        alt_start_width=1,
        alt_start_depth=1,
        patience=int(args.patience),
        width_expansion_patience=10,
        depth_expansion_patience=2,
        delta=float(args.delta),
        max_epochs=int(args.max_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        max_width=int(args.max_width),
        max_depth=int(args.max_depth),
        max_neurons=int(args.max_neurons),
        width_stage_margin_patience=int(args.width_stage_margin_patience),
        width_stage_min_improve_pct=float(args.width_stage_min_improve_pct),
        use_bn=bool(args.use_bn),
        demo=False,
    )


def run_task_ablation(
    *,
    task_name: str,
    task_root: Path,
    cfg: rg.RunConfig,
    source_run_root: Path,
    architectures: Sequence[Sequence[int]],
    repeat_count: int,
    repeat_index: Optional[int],
    device,
    log: ContinuousLogger,
    batch_controller,
) -> Dict[str, Any]:
    task_batch_size = rg.batch_size_for_task(task_name, cfg.batch_size)
    task = build_task(task_name, cfg.data_dir, task_batch_size, cfg.num_workers, cfg.seed)
    source_summary = load_source_task_summary(source_run_root, task_name)

    ablation_runs: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    source_adp_runs = list(source_summary.get("adp_runs", []))
    source_paired_stl_runs = list(source_summary.get("paired_stl_runs", []))
    source_refs = [
        ("adp", entry.get("phase"), entry.get("architecture"), float(entry.get("best_val", float("inf"))))
        for entry in source_adp_runs
    ] + [
        ("paired_stl", entry.get("phase"), entry.get("architecture"), float(entry.get("best_val", float("inf"))))
        for entry in source_paired_stl_runs
    ]

    best_ablation: Optional[Dict[str, Any]] = None
    repeat_count = max(1, int(repeat_count))
    curve_rows: List[Dict[str, Any]] = []
    repeat_indices = [int(repeat_index)] if repeat_index is not None else list(range(1, repeat_count + 1))

    for repeat_id in repeat_indices:
        for architecture in architectures:
            phase_name = phase_name_for_architecture(architecture, repeat_id)
            log.log_console(
                f"[ABLATION:{task_name}] STL phase start: {phase_name} architecture={rg.format_architecture_for_report(architecture)}"
            )
            summary = rg.run_stl_phase(
                task,
                task_root,
                cfg,
                device,
                list(architecture),
                phase_name=phase_name,
                source_phase=None,
                batch_controller=batch_controller,
            )
            ablation_runs.append(summary)
            parameter_count = parameter_count_for_summary(task, task_root, summary)

            curve_rows.append(
                {
                    "task": task_name,
                    "repeat": int(repeat_id),
                    "phase": phase_name,
                    "architecture": rg.format_architecture_for_report(architecture),
                    "parameter_count": int(parameter_count),
                    "best_val": float(summary.get("best_val", float("inf"))),
                    "best_epoch": int(summary.get("best_epoch", 0)),
                    "final_epoch": int(summary.get("final_epoch", summary.get("best_epoch", 0))),
                    "test_loss": (summary.get("test_metrics") or {}).get("test_loss"),
                    "test_acc": (summary.get("test_metrics") or {}).get("test_acc"),
                }
            )

            if best_ablation is None or float(summary.get("best_val", float("inf"))) < float(best_ablation.get("best_val", float("inf"))):
                best_ablation = {
                    **summary,
                    "repeat": int(repeat_id),
                    "parameter_count": int(parameter_count),
                }

            rows.append(
                {
                    "task": task_name,
                    "repeat": int(repeat_id),
                    "row_type": "ablation_stl",
                    "phase": phase_name,
                    "architecture": rg.format_architecture_for_report(architecture),
                    "parameter_count": int(parameter_count),
                    "best_val": float(summary.get("best_val", float("inf"))),
                    "best_epoch": int(summary.get("best_epoch", 0)),
                    "final_epoch": int(summary.get("final_epoch", summary.get("best_epoch", 0))),
                    "test_loss": (summary.get("test_metrics") or {}).get("test_loss"),
                    "test_acc": (summary.get("test_metrics") or {}).get("test_acc"),
                }
            )

            for ref_kind, ref_phase, ref_arch, ref_best_val in source_refs:
                comparisons.append(
                    comparison_row(
                        task_name=task_name,
                        repeat_index=repeat_id,
                        ablation_phase=phase_name,
                        ablation_architecture=architecture,
                        ablation_parameter_count=parameter_count,
                        ablation_best_val=float(summary.get("best_val", float("inf"))),
                        ref_phase=str(ref_phase),
                        ref_kind=ref_kind,
                        ref_architecture=ref_arch,
                        ref_best_val=float(ref_best_val),
                    )
                )

    rg.write_csv(
        task_root / "ablation_summary.csv",
        rows,
        fieldnames=["task", "repeat", "row_type", "phase", "architecture", "parameter_count", "best_val", "best_epoch", "final_epoch", "test_loss", "test_acc"],
    )
    rg.write_json(
        task_root / "ablation_summary.json",
        {
            "task": task_name,
            "source_task_summary": str(source_run_root / task_name / "task_summary.json"),
            "source_adp_runs": source_adp_runs,
            "source_paired_stl_runs": source_paired_stl_runs,
            "ablation_stl_runs": ablation_runs,
            "comparisons": comparisons,
            "best_ablation": best_ablation,
            "repeat_count": repeat_count,
            "architecture_count": len(architectures),
        },
    )

    if curve_rows:
        plot_task_ablation(task_root, task_name, curve_rows)

    return {
        "task": task_name,
        "source_adp_runs": source_adp_runs,
        "source_paired_stl_runs": source_paired_stl_runs,
        "ablation_stl_runs": ablation_runs,
        "comparisons": comparisons,
        "best_ablation": best_ablation,
        "repeat_count": repeat_count,
        "curve_rows": curve_rows,
    }


def plot_task_ablation(task_root: Path, task_name: str, rows: Sequence[Dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(18, 12))
    repeats = sorted({int(row["repeat"]) for row in rows})
    cmap = plt.get_cmap("tab10")
    for idx, repeat in enumerate(repeats):
        repeat_rows = [row for row in rows if int(row["repeat"]) == repeat]
        repeat_rows = sorted(repeat_rows, key=lambda row: (float(row["parameter_count"]), float(row["best_val"])))
        if not repeat_rows:
            continue
        xs = [float(row["parameter_count"]) for row in repeat_rows]
        ys = [float(row["best_val"]) for row in repeat_rows]
        color = cmap(idx % 10)
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.8, label=f"repeat {repeat}")
        ax.scatter(xs, ys, color=color, s=8, alpha=0.35)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("Best validation loss (log scale)")
    ax.set_title(f"{task_name}: STL ablation loss vs parameters", fontsize=16, pad=16)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    plot_path = task_root / "ablation_loss_vs_params.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    rg.write_json(
        task_root / "ablation_plot.json",
        {
            "task": task_name,
            "plot_path": str(plot_path),
            "repeats": repeats,
            "note": "Each repeat line is one full architecture sweep over widths 16..1024 in steps of 16 and depths 1..10.",
        },
    )
    return plot_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full STL architecture ablation for selected tabular DAE/DNN tasks.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default=None)
    p.add_argument("--source-run-root", default="MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--batch-size", type=int, default=32768)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-width", type=int, default=1024)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--width-stage-margin-patience", type=int, default=10)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--min-width", type=int, default=DEFAULT_MIN_WIDTH)
    p.add_argument("--width-step", type=int, default=DEFAULT_WIDTH_STEP)
    p.add_argument("--min-depth", type=int, default=DEFAULT_MIN_DEPTH)
    p.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
    p.add_argument("--repeat-index", type=int, default=None, help="Run exactly one repeat index, useful for parallel fan-out.")
    p.add_argument("--stl-width", type=int, default=128)
    p.add_argument("--stl-depth", type=int, default=2)
    p.add_argument("--use-bn", action="store_true", default=True)
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    p.add_argument("--widths", default="")
    p.add_argument("--depths", default="")
    p.add_argument("--architecture", action="append", default=[], help="Explicit hidden widths, e.g. --architecture 64,64,64")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [str(t).lower() for t in args.tasks]
    architectures = build_architectures(args)
    if not architectures:
        raise SystemExit("No architectures requested.")

    source_run_root = Path(args.source_run_root)
    run_root = Path(args.run_root) if args.run_root else Path(args.results_dir) / f"stl_ablation_{rg.now_stamp()}"
    run_root.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg(args, tasks, run_root)
    rg.seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_controller = None

    logger = ContinuousLogger(run_root, "stl_ablation", "stl_ablation")
    logger.log_console(f"Run root: {run_root}")
    logger.log_console(f"Tasks: {tasks}")
    logger.log_console(f"Architectures: {[rg.format_architecture_for_report(a) for a in architectures]}")
    logger.log_console(f"Repeat count: {int(args.repeat_count)}")
    logger.log_console(f"Source run root: {source_run_root}")
    logger.log_console(f"Device: {device}")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    task_reports: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    for task_name in tasks:
        task_root = run_root / task_name
        task_root.mkdir(parents=True, exist_ok=True)
        report = run_task_ablation(
            task_name=task_name,
            task_root=task_root,
            cfg=cfg,
            source_run_root=source_run_root,
            architectures=architectures,
            repeat_count=int(args.repeat_count),
            repeat_index=args.repeat_index,
            device=device,
            log=logger,
            batch_controller=batch_controller,
        )
        task_reports.append(report)
        comparison_rows.extend(report["comparisons"])

    if comparison_rows:
        rg.write_csv(
            run_root / "comparison_summary.csv",
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
        run_root / "comparison_summary.json",
        {
            "tasks": tasks,
            "architectures": architectures,
            "source_run_root": str(source_run_root),
            "repeat_count": int(args.repeat_count),
            "reports": task_reports,
        },
    )
    logger.close()


if __name__ == "__main__":
    main()
