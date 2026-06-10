from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

import run_goliath as rg


DEFAULT_DEPTHS = [3, 4, 6, 8, 10]
DEFAULT_WIDTHS = [64, 96, 128, 160, 192, 224, 256]
DEFAULT_TASKS = ["simulation", "prediction"]


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    rg.write_csv(path, rows, fieldnames)


def parse_int_list(values: Sequence[str]) -> List[int]:
    return [int(v) for v in values]


def phase_label(depth: int, width: int) -> str:
    return f"stl_ablation_d{int(depth):02d}_w{int(width):03d}_{'_'.join(str(int(width)) for _ in range(int(depth)))}"


def candidate_dir_for(task_root: Path, depth: int, width: int) -> Path:
    return task_root / "stl_ablation" / f"d{int(depth):02d}" / f"w{int(width):03d}" / "cand_000"


def task_rows(task: str, task_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in task_summary.get("candidates", []):
        test_metrics = entry.get("test_metrics") or {}
        rows.append(
            {
                "task": task,
                "suite": "small_grid",
                "depth": entry.get("depth"),
                "width": entry.get("width"),
                "phase": entry.get("phase"),
                "architecture": rg.format_architecture_for_report(entry.get("architecture")),
                "params": entry.get("params"),
                "best_val": entry.get("best_val"),
                "best_epoch": entry.get("best_epoch"),
                "final_epoch": entry.get("final_epoch"),
                "test_loss": test_metrics.get("test_loss"),
                "test_acc": test_metrics.get("test_acc"),
                "knn_acc": test_metrics.get("knn_acc"),
                "cluster_nmi": test_metrics.get("cluster_nmi"),
                "candidate_dir": entry.get("candidate_dir"),
                "resumed": entry.get("resumed", False),
            }
        )
    return rows


def run_one_candidate(
    *,
    task,
    task_root: Path,
    cfg: rg.RunConfig,
    device,
    depth: int,
    width: int,
) -> Dict[str, Any]:
    label = phase_label(depth, width)
    candidate_dir = candidate_dir_for(task_root, depth, width)
    candidate_state = rg.load_json_if_exists(candidate_dir / "candidate_state.json") or {}
    already_completed = bool(candidate_state.get("completed", False)) and (candidate_dir / "checkpoint_last.pt").exists()

    architecture = [int(width) for _ in range(int(depth))]
    model = rg.make_stl_model(task, architecture, cfg.use_bn).to(device)
    logger = rg.ContinuousLogger(candidate_dir, f"{task.name}_{label}", label, resume=(candidate_dir / "checkpoint_last.pt").exists())

    if already_completed:
        loaded_model, _, ckpt = rg.load_candidate_model(candidate_dir, device)
        test_metrics = rg.eval_final(loaded_model, task, device, reconstruct=rg.task_reconstruct(task))
        logger.close()
        return {
            "depth": int(depth),
            "width": int(width),
            "phase": label,
            "architecture": [int(v) for v in architecture],
            "params": rg.count_model_parameters(model),
            "best_val": float(ckpt.get("best_val", float("inf"))),
            "best_epoch": int(ckpt.get("best_epoch", 0)),
            "final_epoch": int(ckpt.get("epoch", 0)),
            "test_metrics": test_metrics,
            "candidate_dir": str(candidate_dir),
            "resumed": True,
        }

    result = rg.training_loop(
        task=task,
        model=model,
        candidate_dir=candidate_dir,
        cfg=cfg,
        device=device,
        logger=logger,
        reconstruct=rg.task_reconstruct(task),
        resume=True,
    )
    test_metrics = rg.eval_final(model, task, device, reconstruct=rg.task_reconstruct(task))
    logger.close()
    return {
        "depth": int(depth),
        "width": int(width),
        "phase": label,
        "architecture": [int(v) for v in architecture],
        "params": rg.count_model_parameters(model),
        "best_val": float(result.best_val),
        "best_epoch": int(result.best_epoch),
        "final_epoch": int(result.final_epoch),
        "test_metrics": test_metrics,
        "candidate_dir": str(candidate_dir),
        "resumed": False,
    }


def run_task(
    *,
    task,
    task_root: Path,
    cfg: rg.RunConfig,
    device,
    depths: Sequence[int],
    widths: Sequence[int],
) -> Dict[str, Any]:
    task_state_path = task_root / "task_state.json"
    task_summary_path = task_root / "task_summary.json"
    existing_summary = rg.load_json_if_exists(task_summary_path) or {}
    existing_state = rg.load_json_if_exists(task_state_path) or {}
    if bool(existing_state.get("completed", False)) and existing_summary.get("candidates"):
        return existing_summary

    task_root.mkdir(parents=True, exist_ok=True)
    task_state: Dict[str, Any] = {
        "task": task.name,
        "suite": "small_grid",
        "depths": list(int(v) for v in depths),
        "widths": list(int(v) for v in widths),
        "completed_candidates": list(existing_state.get("completed_candidates", [])),
        "next_candidate_index": int(existing_state.get("next_candidate_index", 0)),
        "completed": False,
    }

    candidate_rows: List[Dict[str, Any]] = []
    candidate_index = 0
    for depth in depths:
        for width in widths:
            row = run_one_candidate(task=task, task_root=task_root, cfg=cfg, device=device, depth=int(depth), width=int(width))
            row["candidate_index"] = candidate_index
            candidate_rows.append(row)
            candidate_index += 1
            task_state["completed_candidates"] = [r["phase"] for r in candidate_rows if r.get("candidate_dir")]
            task_state["next_candidate_index"] = candidate_index
            write_json(task_state_path, task_state)
            write_json(task_summary_path, {"task": task.name, "suite": "small_grid", "candidates": candidate_rows})

    task_state["completed"] = True
    write_json(task_state_path, task_state)
    task_summary = {
        "task": task.name,
        "suite": "small_grid",
        "depths": list(int(v) for v in depths),
        "widths": list(int(v) for v in widths),
        "candidates": candidate_rows,
        "best_candidate": min(candidate_rows, key=lambda row: float(row["best_val"])) if candidate_rows else None,
    }
    write_json(task_summary_path, task_summary)
    write_csv(
        task_root / "task_summary.csv",
        task_rows(task.name, task_summary),
        [
            "task",
            "suite",
            "depth",
            "width",
            "phase",
            "architecture",
            "params",
            "best_val",
            "best_epoch",
            "final_epoch",
            "test_loss",
            "test_acc",
            "knn_acc",
            "cluster_nmi",
            "candidate_dir",
            "resumed",
        ],
    )
    return task_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the lightweight no-repeat STL grid used by the archived small study.")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", type=str, default=None)
    p.add_argument("--source-run-root", type=str, default="MLPS/tabular/shared/dae_dnn/results/archive/classification_trial1")
    p.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    p.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    p.add_argument("--batch-size", type=int, default=9312)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-bn", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--demo-tasks", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [t.lower() for t in args.tasks]
    if args.demo:
        tasks = tasks[: max(1, int(args.demo_tasks))]
        args.max_epochs = min(int(args.max_epochs), 1)
        args.patience = min(int(args.patience), 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_root = Path(args.run_root) if args.run_root else Path(args.results_dir) / "stl" / "small_grid" / f"small_grid_{now_stamp()}"
    run_root.mkdir(parents=True, exist_ok=True)

    cfg = rg.RunConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        run_root=str(run_root),
        tasks=list(tasks),
        phases=["stl_ablation"],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        stl_width=max(int(w) for w in args.widths),
        stl_depth=max(int(d) for d in args.depths),
        alt_start_width=1,
        alt_start_depth=1,
        patience=int(args.patience),
        width_expansion_patience=10,
        depth_expansion_patience=2,
        delta=1e-4,
        max_epochs=int(args.max_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        max_width=max(int(w) for w in args.widths),
        max_depth=max(int(d) for d in args.depths),
        max_neurons=10_000_000,
        width_stage_margin_patience=0,
        width_stage_min_improve_pct=1.0,
        use_bn=not bool(args.no_bn),
        demo=bool(args.demo),
        metrics_every=0,
        min_width=min(int(w) for w in args.widths),
        width_step=1,
        width_count_per_depth=len(args.widths),
        parameter_matched=False,
    )

    write_json(
        run_root / "run_metadata.json",
        {
            "config": asdict(cfg),
            "git_commit": rg.git_commit(),
            "device": str(device),
            "tasks": tasks,
            "depths": list(int(v) for v in args.depths),
            "widths": list(int(v) for v in args.widths),
            "source_run_root": args.source_run_root,
            "suite": "small_grid",
            "timestamp": now_stamp(),
        },
    )

    progress_path = run_root / "run_progress.csv"
    logger = rg.ContinuousLogger(run_root, "stl_small_grid", "stl_small_grid", resume=progress_path.exists())
    logger.log_console(f"Run root: {run_root}")
    logger.log_console(f"Tasks: {tasks}")
    logger.log_console(f"Depths: {list(int(v) for v in args.depths)}")
    logger.log_console(f"Widths: {list(int(v) for v in args.widths)}")
    logger.log_console(f"Source root: {args.source_run_root}")
    logger.log_console(f"Device: {device}")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    task_summaries: Dict[str, Dict[str, Any]] = {}
    try:
        for task_name in tasks:
            task_batch_size = rg.batch_size_for_task(task_name, int(args.batch_size))
            task = rg.build_task(task_name, args.data_dir, task_batch_size, int(args.num_workers), int(args.seed))
            rg.refresh_task_loaders(task, task_batch_size)
            task_root = run_root / task_name
            task_root.mkdir(parents=True, exist_ok=True)
            write_json(
                task_root / "task_metadata.json",
                {
                    "task": task.name,
                    "in_dim": task.in_dim,
                    "out_dim": task.out_dim,
                    "task_type": task.task_type,
                    "extra": task.extra,
                    "config": asdict(cfg),
                    "batch_size": int(task_batch_size),
                    "suite": "small_grid",
                    "source_run_root": args.source_run_root,
                },
            )
            task_summary = run_task(
                task=task,
                task_root=task_root,
                cfg=cfg,
                device=device,
                depths=list(int(v) for v in args.depths),
                widths=list(int(v) for v in args.widths),
            )
            task_summaries[task_name] = task_summary
            write_json(task_root / "task_summary.json", task_summary)
            logger.log_console(f"[TASK] done {task_name}")
    finally:
        logger.close()

    final_report = {
        "run_root": str(run_root),
        "git_commit": rg.git_commit(),
        "device": str(device),
        "suite": "small_grid",
        "config": asdict(cfg),
        "source_run_root": args.source_run_root,
        "summary": {
            "tasks_requested": list(tasks),
            "num_tasks_requested": len(tasks),
            "tasks_completed": [name for name, summary in task_summaries.items() if summary.get("candidates")],
            "num_tasks_completed": sum(1 for summary in task_summaries.values() if summary.get("candidates")),
        },
        "tasks": [],
    }
    for task_name in tasks:
        summary = task_summaries.get(task_name)
        if not summary:
            continue
        best = summary.get("best_candidate") or {}
        final_report["tasks"].append(
            {
                "task": task_name,
                "best_candidate": best,
                "candidate_count": len(summary.get("candidates", [])),
                "depths": summary.get("depths", []),
                "widths": summary.get("widths", []),
            }
        )
    write_json(run_root / "final_report.json", final_report)
    write_text(
        run_root / "final_report.md",
        "# Small STL Grid Final Report\n\n"
        f"- Run root: `{run_root}`\n"
        f"- Suite: `small_grid`\n"
        f"- Tasks: `{tasks}`\n"
        f"- Depths: `{list(int(v) for v in args.depths)}`\n"
        f"- Widths: `{list(int(v) for v in args.widths)}`\n"
        f"- Source run root: `{args.source_run_root}`\n",
    )


if __name__ == "__main__":
    main()
