from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Sequence, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

from MLPS.tabular.shared.dae_dnn.tasks import task_names
from utils.adp_logging import ContinuousLogger

try:  # pragma: no cover - import shim for direct script execution
    import run_goliath as rg
    import run_adp_w2d_suite_parallel as suite
except ModuleNotFoundError:  # pragma: no cover - import shim for package-style imports
    from MLPS.tabular.shared.dae_dnn import run_goliath as rg
    from MLPS.tabular.shared.dae_dnn import run_adp_w2d_suite_parallel as suite


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interleaved parallel ADP width-to-depth suite launcher that keeps workers full across repeats."
    )
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default=None)
    p.add_argument("--tasks", nargs="+", default=list(suite.DEFAULT_TASKS))
    p.add_argument("--batch-size", type=int, default=9312)
    p.add_argument("--hidden", type=int, nargs="+", default=[1])
    p.add_argument("--adp-mode", default="width_to_depth", choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"])
    p.add_argument("--mode", default="adp", choices=["adp"])
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--trials-width", type=int, default=10)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=1)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10000000)
    p.add_argument("--width-stage-margin-patience", type=int, default=10)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--metrics-interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--repeat-count", type=int, default=4)
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=False)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.add_argument("--dry-run", action="store_true", help="Print the pending job order and exit without launching workers.")
    return p.parse_args()


def normalize_tasks(raw_tasks: Sequence[str]) -> List[str]:
    tasks = [str(t).lower() for t in raw_tasks]
    if "all" in tasks:
        return list(suite.DEFAULT_TASKS)
    allowed = set(task_names()) | set(suite.DEFAULT_TASKS)
    return [task for task in tasks if task in allowed]


def build_pending_jobs(args: argparse.Namespace, tasks: Sequence[str], run_root: Path) -> Deque[Tuple[int, str, Path]]:
    pending: Deque[Tuple[int, str, Path]] = deque()
    max_repeat_count = max(suite.repeat_count_for_task(task, int(args.repeat_count)) for task in tasks)
    for repeat_index in range(1, max_repeat_count + 1):
        repeat_root = run_root / f"repeat_{repeat_index:02d}"
        for task_name in tasks:
            if repeat_index <= suite.repeat_count_for_task(task_name, int(args.repeat_count)):
                pending.append((repeat_index, task_name, repeat_root / task_name))
    return pending


def job_is_done(task_root: Path) -> bool:
    return suite.task_completed(task_root)


def main() -> None:
    args = parse_args()
    tasks = normalize_tasks(args.tasks)
    if not tasks:
        raise SystemExit("No tasks requested.")

    run_root = Path(args.run_root) if args.run_root else Path(args.results_dir) / "adp" / "w2d" / "repeat4_plus_sim_pred5_v1"
    run_root.mkdir(parents=True, exist_ok=True)
    resume_logs = (run_root / "training_log.txt").exists() or (run_root / "training_stats.csv").exists()
    logger = ContinuousLogger(run_root, "adp_w2d_suite_interleaved", "adp_width_to_depth_interleaved", resume=resume_logs)
    reports: List[Dict[str, Any]] = []
    active: Dict[subprocess.Popen[Any], Tuple[int, str, Path, List[str]]] = {}

    pending = build_pending_jobs(args, tasks, run_root)
    logger.log_console(f"Run root: {run_root}")
    logger.log_console(f"Tasks: {tasks}")
    logger.log_console(f"Batch size: {int(args.batch_size)}")
    logger.log_console(f"Hidden seed: {list(map(int, args.hidden))}")
    logger.log_console(
        "Repeat counts: " + ", ".join(f"{task}={suite.repeat_count_for_task(task, int(args.repeat_count))}" for task in tasks)
    )
    logger.log_console(f"Concurrency: {int(args.concurrency)}")
    logger.log_console(f"Interleaved repeats: true")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    if args.dry_run:
        for repeat_index, task_name, task_root in pending:
            state = "completed" if job_is_done(task_root) else "pending"
            logger.log_console(f"[PLAN] repeat={repeat_index} task={task_name} state={state} root={task_root}")
        logger.close()
        return

    try:
        while pending or active:
            while pending and len(active) < int(args.concurrency):
                repeat_index, task_name, task_root = pending.popleft()
                if job_is_done(task_root):
                    reports.append(
                        {
                            "repeat_index": int(repeat_index),
                            "task": task_name,
                            "status": "completed",
                            "task_root": str(task_root),
                            "skipped": True,
                        }
                    )
                    continue
                cmd = suite.build_worker_command(args, task_name, task_root)
                task_root.mkdir(parents=True, exist_ok=True)
                proc = subprocess.Popen(cmd)
                active[proc] = (repeat_index, task_name, task_root, cmd)
                logger.log_console(f"[TASK] launched repeat={repeat_index} task={task_name}")

            if not active:
                continue

            finished: List[subprocess.Popen[Any]] = []
            for proc in list(active):
                code = proc.poll()
                if code is None:
                    continue
                repeat_index, task_name, task_root, cmd = active[proc]
                suite.terminate_child_process(proc)
                finished.append(proc)
                if code != 0:
                    logger.log_console(f"[TASK] failed repeat={repeat_index} task={task_name} code={code}")
                    reports.append(
                        {
                            "repeat_index": int(repeat_index),
                            "task": task_name,
                            "status": "failed",
                            "task_root": str(task_root),
                            "exit_code": int(code),
                            "command": cmd,
                        }
                    )
                    pending.appendleft((repeat_index, task_name, task_root))
                else:
                    logger.log_console(f"[TASK] completed repeat={repeat_index} task={task_name}")
                    reports.append(
                        {
                            "repeat_index": int(repeat_index),
                            "task": task_name,
                            "status": "completed",
                            "task_root": str(task_root),
                            "command": cmd,
                        }
                    )
            for proc in finished:
                active.pop(proc, None)

            if active:
                time.sleep(2)
    finally:
        for proc in list(active):
            suite.terminate_child_process(proc)
        summary = {
            "run_root": str(run_root),
            "tasks": tasks,
            "batch_size": int(args.batch_size),
            "hidden": [int(v) for v in args.hidden],
            "adp_mode": str(args.adp_mode),
            "repeat_count": int(args.repeat_count),
            "repeat_counts": {task: suite.repeat_count_for_task(task, int(args.repeat_count)) for task in tasks},
            "interleaved_repeats": True,
            "reports": reports,
        }
        (run_root / "suite_summary_interleaved.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        logger.close()


if __name__ == "__main__":
    main()
