from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

import run_goliath as rg
import run_stl_ablation as stl
from MLPS.tabular.shared.dae_dnn.tasks import build_task
from utils.adp_logging import ContinuousLogger


DEFAULT_TASKS = [
    "classification",
    "autoencoding",
    "generation",
    "denoising",
    "anomaly",
    "simulation",
]


def parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark one complete STL epoch at the task-specific max widths.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default=None)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--depths", default="1,2,3,4,5")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--delta", type=float, default=1e-4)
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
    p.add_argument("--stl-width", type=int, default=128)
    p.add_argument("--stl-depth", type=int, default=2)
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=False)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.add_argument("--use-bn", action="store_true", default=True)
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    return p.parse_args()


def max_width_for_task(task_name: str, depth: int) -> int:
    name = str(task_name).lower()
    if name == "classification":
        table = stl.REPRESENTATION_ONLY_MAX_WIDTH_BY_DEPTH
    elif name in {"autoencoding", "generation", "denoising"}:
        table = stl.REPRESENTATION_MAX_WIDTH_BY_DEPTH
    elif name == "anomaly":
        table = stl.ANOMALY_MAX_WIDTH_BY_DEPTH
    elif name in {"simulation", "prediction"}:
        table = stl.SIMULATION_MAX_WIDTH_BY_DEPTH
    else:
        raise ValueError(f"Unsupported task for max-width benchmark: {task_name}")
    return int(table[int(depth)])


def cleanup_runtime() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def build_task_for_benchmark(
    *,
    task_name: str,
    data_dir: str,
    batch_size_override: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
):
    probe_task = build_task(task_name, data_dir, 16, num_workers, seed, pin_memory=pin_memory)
    actual_batch_size = stl.stl_batch_size_for_task(task_name, probe_task, batch_size_override)
    if actual_batch_size != 16:
        probe_task = build_task(task_name, data_dir, actual_batch_size, num_workers, seed, pin_memory=pin_memory)
    return probe_task, int(actual_batch_size)


def make_cfg(args: argparse.Namespace, task_name: str, run_root: Path) -> rg.RunConfig:
    return rg.RunConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        run_root=str(run_root),
        tasks=[task_name],
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
        max_epochs=1,
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
        min_width=int(args.min_width),
        width_step=int(args.width_step),
        parameter_matched=True,
    )


def write_markdown_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    grouped: Dict[tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task"]), int(row["depth"])), []).append(row)
    lines = ["# STL Epoch Timing Benchmark", ""]
    for (task_name, depth) in sorted(grouped.keys()):
        task_rows = grouped[(task_name, depth)]
        success_rows = [row for row in task_rows if row["status"] == "ok"]
        lines.append(f"## {task_name} depth {depth}")
        lines.append("")
        if success_rows:
            avg_seconds = statistics.mean(float(row["elapsed_seconds"]) for row in success_rows)
            width = int(success_rows[0]["width"])
            batch_size = int(success_rows[0]["batch_size"])
            lines.append(f"- width: `{width}`")
            lines.append(f"- batch size: `{batch_size}`")
            lines.append(f"- successful repeats: `{len(success_rows)}`")
            lines.append(f"- average epoch seconds: `{avg_seconds:.3f}`")
        else:
            lines.append("- no successful repeats")
        failed_rows = [row for row in task_rows if row["status"] != "ok"]
        if failed_rows:
            for row in failed_rows:
                lines.append(f"- failure repeat `{row['repeat']}`: `{row['error_type']}` `{row['error_message']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rg.seed_everything(int(args.seed))
    depths = [int(v) for v in parse_csv_ints(args.depths)]
    tasks = [str(task).lower() for task in args.tasks]
    run_root = Path(args.run_root) if args.run_root else Path(args.results_dir) / f"epoch_time_benchmark_{rg.now_stamp()}"
    run_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master_log = ContinuousLogger(run_root, "benchmark_stl_epoch_times", "epoch_timing")
    master_log.log_console(f"Run root: {run_root}")
    master_log.log_console(f"Tasks: {tasks}")
    master_log.log_console(f"Depths: {depths}")
    master_log.log_console(f"Repeats: {int(args.repeats)}")
    master_log.log_console(f"Device: {device}")

    for task_name in tasks:
        for depth in depths:
            width = max_width_for_task(task_name, depth)
            architecture = [int(width) for _ in range(int(depth))]
            for repeat_idx in range(1, int(args.repeats) + 1):
                cleanup_runtime()
                task, batch_size = build_task_for_benchmark(
                    task_name=task_name,
                    data_dir=str(args.data_dir),
                    batch_size_override=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    seed=int(args.seed),
                    pin_memory=bool(args.pin_memory),
                )
                model = rg.make_stl_model(task, architecture, bool(args.use_bn)).to(device)
                cfg = make_cfg(args, task_name, run_root)
                candidate_dir = run_root / task_name / f"d{depth}_w{width}" / f"repeat_{repeat_idx:02d}"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                logger = ContinuousLogger(candidate_dir, f"{task_name}_d{depth}_w{width}_r{repeat_idx:02d}", "epoch_timing")
                params = int(rg.count_model_parameters(model))
                logger.log_console(
                    f"[BENCH] task={task_name} depth={depth} width={width} repeat={repeat_idx} batch_size={batch_size} parameters={params}"
                )
                started = time.perf_counter()
                row: Dict[str, Any] = {
                    "task": task_name,
                    "depth": int(depth),
                    "width": int(width),
                    "repeat": int(repeat_idx),
                    "batch_size": int(batch_size),
                    "parameters": int(params),
                    "status": "ok",
                    "elapsed_seconds": None,
                    "best_val": None,
                    "best_epoch": None,
                    "final_epoch": None,
                    "error_type": "",
                    "error_message": "",
                }
                try:
                    result = rg.training_loop(
                        task=task,
                        model=model,
                        candidate_dir=candidate_dir,
                        cfg=cfg,
                        device=device,
                        logger=logger,
                        reconstruct=(task.task_type == "reconstruction"),
                        resume=False,
                        batch_controller=None,
                        display_best_floor=None,
                    )
                    row["elapsed_seconds"] = float(time.perf_counter() - started)
                    row["best_val"] = float(result.best_val)
                    row["best_epoch"] = int(result.best_epoch)
                    row["final_epoch"] = int(result.final_epoch)
                    logger.log_console(
                        f"[BENCH:DONE] task={task_name} depth={depth} width={width} repeat={repeat_idx} elapsed_seconds={row['elapsed_seconds']:.3f}"
                    )
                except Exception as exc:
                    row["status"] = "error"
                    row["elapsed_seconds"] = float(time.perf_counter() - started)
                    row["error_type"] = type(exc).__name__
                    row["error_message"] = str(exc)
                    logger.log_console(
                        f"[BENCH:FAIL] task={task_name} depth={depth} width={width} repeat={repeat_idx} "
                        f"error_type={row['error_type']} error_message={row['error_message']}"
                    )
                finally:
                    logger.close()
                    summary_rows.append(row)
                    cleanup_runtime()

    rg.write_json(
        run_root / "epoch_time_summary.json",
        {
            "tasks": tasks,
            "depths": depths,
            "repeats": int(args.repeats),
            "rows": summary_rows,
        },
    )
    rg.write_csv(
        run_root / "epoch_time_summary.csv",
        summary_rows,
        fieldnames=[
            "task",
            "depth",
            "width",
            "repeat",
            "batch_size",
            "parameters",
            "status",
            "elapsed_seconds",
            "best_val",
            "best_epoch",
            "final_epoch",
            "error_type",
            "error_message",
        ],
    )
    write_markdown_summary(run_root / "epoch_time_summary.md", summary_rows)
    master_log.close()


if __name__ == "__main__":
    main()
