from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch

from DAE.DNN.tasks import build_task
from utils.adp_logging import ContinuousLogger

import run_goliath as rg
import run_stl_ablation as stl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel STL ablation launcher with per-architecture child run roots.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default=None)
    p.add_argument("--source-run-root", default="MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current")
    p.add_argument("--tasks", nargs="+", default=list(stl.DEFAULT_TASKS))
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
    p.add_argument("--min-width", type=int, default=16)
    p.add_argument("--width-step", type=int, default=16)
    p.add_argument("--min-depth", type=int, default=1)
    p.add_argument("--repeat-count", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--stl-width", type=int, default=128)
    p.add_argument("--stl-depth", type=int, default=2)
    p.add_argument("--use-bn", action="store_true", default=True)
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    return p.parse_args()


def build_worker_command(
    *,
    args: argparse.Namespace,
    task_name: str,
    architecture: Sequence[int],
    repeat_index: int,
    child_run_root: Path,
) -> List[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_stl_ablation.py"),
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(args.results_dir),
        "--run-root",
        str(child_run_root),
        "--source-run-root",
        str(args.source_run_root),
        "--tasks",
        task_name,
        "--architecture",
        ",".join(str(int(v)) for v in architecture),
        "--repeat-index",
        str(int(repeat_index)),
        "--repeat-count",
        "1",
        "--batch-size",
        str(int(args.batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--seed",
        str(int(args.seed)),
        "--patience",
        str(int(args.patience)),
        "--delta",
        str(float(args.delta)),
        "--max-epochs",
        str(int(args.max_epochs)),
        "--lr",
        str(float(args.lr)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--grad-clip",
        str(float(args.grad_clip)),
        "--max-width",
        str(int(args.max_width)),
        "--max-depth",
        str(int(args.max_depth)),
        "--max-neurons",
        str(int(args.max_neurons)),
        "--width-stage-margin-patience",
        str(int(args.width_stage_margin_patience)),
        "--width-stage-min-improve-pct",
        str(float(args.width_stage_min_improve_pct)),
        "--min-width",
        str(int(args.min_width)),
        "--width-step",
        str(int(args.width_step)),
        "--min-depth",
        str(int(args.min_depth)),
        "--stl-width",
        str(int(args.stl_width)),
        "--stl-depth",
        str(int(args.stl_depth)),
    ] + (["--no-bn"] if not bool(args.use_bn) else [])


def child_summary_path(child_run_root: Path, task_name: str) -> Path:
    return child_run_root / task_name / "ablation_summary.json"


def child_completed(child_run_root: Path, task_name: str) -> bool:
    path = child_summary_path(child_run_root, task_name)
    data = rg.load_json_if_exists(path) or {}
    return bool(data.get("ablation_stl_runs"))


def load_task_child_summary(child_run_root: Path, task_name: str) -> Dict[str, Any]:
    path = child_summary_path(child_run_root, task_name)
    data = rg.load_json_if_exists(path)
    if not isinstance(data, dict):
        raise FileNotFoundError(f"Missing child summary: {path}")
    return data


def aggregate_task(task_name: str, task_root: Path, child_roots: Sequence[Path]) -> Dict[str, Any]:
    ablation_runs: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    best_ablation: Optional[Dict[str, Any]] = None
    source_adp_runs: List[Dict[str, Any]] = []
    source_paired_stl_runs: List[Dict[str, Any]] = []
    architecture_keys: set[Tuple[int, ...]] = set()

    for child_root in child_roots:
        summary = load_task_child_summary(child_root, task_name)
        source_adp_runs = list(summary.get("source_adp_runs", source_adp_runs))
        source_paired_stl_runs = list(summary.get("source_paired_stl_runs", source_paired_stl_runs))
        ablation_runs.extend(summary.get("ablation_stl_runs", []))
        comparisons.extend(summary.get("comparisons", []))
        if summary.get("ablation_stl_runs"):
            for entry in summary["ablation_stl_runs"]:
                architecture = [int(v) for v in entry["architecture"]]
                architecture_keys.add(tuple(architecture))
                candidate_dir = Path(entry["checkpoint_best"]).parent
                metadata = rg.load_json_if_exists(candidate_dir / "metadata.json") or {}
                model_cfg = metadata.get("model") or {}
                repeat_token = child_root.parent.name
                repeat_index = int(repeat_token[1:]) if repeat_token.startswith("r") else 1
                curve_rows.append(
                    {
                        "task": task_name,
                        "repeat": int(repeat_index),
                        "phase": str(entry["phase"]),
                        "architecture": rg.format_architecture_for_report(architecture),
                        "parameter_count": int(
                            rg.count_model_parameters(
                                rg.make_model(
                                    int(model_cfg.get("in_dim", 1)),
                                    architecture,
                                    int(model_cfg.get("out_dim", 1)),
                                    bool(model_cfg.get("use_bn", True)),
                                )
                            )
                        ),
                        "best_val": float(entry.get("best_val", float("inf"))),
                        "best_epoch": int(entry.get("best_epoch", 0)),
                        "final_epoch": int(entry.get("final_epoch", entry.get("best_epoch", 0))),
                        "test_loss": (entry.get("test_metrics") or {}).get("test_loss"),
                        "test_acc": (entry.get("test_metrics") or {}).get("test_acc"),
                    }
                )
                if best_ablation is None or float(entry.get("best_val", float("inf"))) < float(best_ablation.get("best_val", float("inf"))):
                    best_ablation = entry

    task_root.mkdir(parents=True, exist_ok=True)
    rg.write_csv(
        task_root / "ablation_summary.csv",
        curve_rows,
        fieldnames=["task", "repeat", "phase", "architecture", "parameter_count", "best_val", "best_epoch", "final_epoch", "test_loss", "test_acc"],
    )
    rg.write_json(
        task_root / "ablation_summary.json",
        {
            "task": task_name,
            "source_adp_runs": source_adp_runs,
            "source_paired_stl_runs": source_paired_stl_runs,
            "ablation_stl_runs": ablation_runs,
            "comparisons": comparisons,
            "best_ablation": best_ablation,
            "repeat_count": len(sorted({int(r["repeat"]) for r in curve_rows})) if curve_rows else 0,
            "architecture_count": len(architecture_keys),
        },
    )
    if curve_rows:
        stl.plot_task_ablation(task_root, task_name, curve_rows)
    return {
        "task": task_name,
        "ablation_stl_runs": ablation_runs,
        "comparisons": comparisons,
        "best_ablation": best_ablation,
        "curve_rows": curve_rows,
    }


def run_parallel_task(args: argparse.Namespace, task_name: str, run_root: Path, architectures: Sequence[Sequence[int]]) -> Dict[str, Any]:
    task_root = run_root / task_name
    child_base = task_root / "_children"
    child_base.mkdir(parents=True, exist_ok=True)
    repeat_indices = list(range(1, int(args.repeat_count) + 1))
    jobs: Deque[Tuple[int, Tuple[int, ...], Path]] = deque()
    for repeat_index in repeat_indices:
        for architecture in architectures:
            phase_name = stl.phase_name_for_architecture(architecture, repeat_index)
            child_root = child_base / f"r{repeat_index:02d}" / phase_name
            jobs.append((repeat_index, tuple(int(v) for v in architecture), child_root))

    active: Dict[subprocess.Popen[Any], Tuple[int, Tuple[int, ...], Path, List[str]]] = {}
    completed_children: List[Path] = []
    launch_count = 0
    while jobs or active:
        while jobs and len(active) < int(args.concurrency):
            repeat_index, architecture, child_root = jobs.popleft()
            if child_completed(child_root, task_name):
                completed_children.append(child_root)
                continue
            cmd = build_worker_command(
                args=args,
                task_name=task_name,
                architecture=architecture,
                repeat_index=repeat_index,
                child_run_root=child_root,
            )
            child_root.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(cmd)
            active[proc] = (repeat_index, architecture, child_root, cmd)
            launch_count += 1

        if not active:
            continue

        finished: List[subprocess.Popen[Any]] = []
        for proc in list(active):
            code = proc.poll()
            if code is None:
                continue
            if code != 0:
                repeat_index, architecture, child_root, cmd = active[proc]
                raise SystemExit(f"Child job failed (repeat={repeat_index}, arch={architecture}, root={child_root}): {' '.join(cmd)}")
            finished.append(proc)

        for proc in finished:
            repeat_index, architecture, child_root, cmd = active.pop(proc)
            completed_children.append(child_root)

        if active:
            time.sleep(2)

    return aggregate_task(task_name, task_root, completed_children)


def main() -> None:
    args = parse_args()
    tasks = [str(t).lower() for t in args.tasks]
    architectures = stl.build_architectures(args)
    if not architectures:
        raise SystemExit("No architectures requested.")

    run_root = Path(args.run_root) if args.run_root else Path(args.results_dir) / f"stl_ablation_parallel_{rg.now_stamp()}"
    run_root.mkdir(parents=True, exist_ok=True)
    logger = ContinuousLogger(run_root, "stl_ablation_parallel", "stl_ablation_parallel")
    logger.log_console(f"Run root: {run_root}")
    logger.log_console(f"Tasks: {tasks}")
    logger.log_console(f"Architectures: {[rg.format_architecture_for_report(a) for a in architectures]}")
    logger.log_console(f"Repeat count: {int(args.repeat_count)}")
    logger.log_console(f"Concurrency: {int(args.concurrency)}")
    logger.log_console(f"Source run root: {args.source_run_root}")
    logger.log_console(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    task_reports: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    for task_name in tasks:
        logger.log_console(f"[TASK] start {task_name}")
        report = run_parallel_task(args, task_name, run_root, architectures)
        task_reports.append(report)
        comparison_rows.extend(report.get("comparisons", []))

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
            "source_run_root": str(args.source_run_root),
            "repeat_count": int(args.repeat_count),
            "concurrency": int(args.concurrency),
            "reports": task_reports,
        },
    )
    logger.close()


if __name__ == "__main__":
    main()
