from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

from MLPS.tabular.shared.dae_dnn.tasks import task_names
import run_goliath as rg
from utils.adp_logging import ContinuousLogger


DEFAULT_TASKS = [
    "classification",
    "autoencoding",
    "generation",
    "denoising",
    "anomaly",
    "simulation",
    "prediction",
]

TASK_REPEAT_COUNTS = {
    "classification": 4,
    "autoencoding": 4,
    "generation": 4,
    "denoising": 4,
    "anomaly": 4,
    "simulation": 5,
    "prediction": 5,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel ADP width-to-depth suite launcher for tabular DNN tasks.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default=None)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--batch-size", type=int, default=9312)
    p.add_argument("--hidden", type=int, nargs="+", default=[1], help="Seed hidden widths for ADP width-to-depth.")
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
    p.add_argument("--concurrency", type=int, default=7)
    p.add_argument(
        "--repeat-count",
        type=int,
        default=4,
        help="Base repeat count; simulation and prediction are given one extra repeat in this suite.",
    )
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=False)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    return p.parse_args()


def task_completed(task_root: Path) -> bool:
    state_path = task_root / "task_state.json"
    if not state_path.exists():
        return False
    try:
        import json

        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(state.get("completed", False)) and not bool(state.get("failed", False))


def repeat_count_for_task(task_name: str, base_repeat_count: int) -> int:
    return int(max(int(base_repeat_count), int(TASK_REPEAT_COUNTS.get(task_name, base_repeat_count))))


def terminate_child_process(proc: subprocess.Popen[Any], timeout_sec: float = 10.0) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    else:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=timeout_sec)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return

    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    else:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=timeout_sec)
    except Exception:
        pass


def build_worker_command(args: argparse.Namespace, task_name: str, task_root: Path) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_task.py"),
        "--task",
        task_name,
        "--mode",
        "adp",
        "--adp-mode",
        str(args.adp_mode),
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(args.results_dir),
        "--run-root",
        str(task_root),
        "--batch-size",
        str(int(args.batch_size)),
        "--max-epochs",
        str(int(args.max_epochs)),
        "--patience",
        str(int(args.patience)),
        "--trials-width",
        str(int(args.trials_width)),
        "--trials-depth",
        str(int(args.trials_depth)),
        "--ex-k",
        str(int(args.ex_k)),
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
        "--metrics-interval",
        str(int(args.metrics_interval)),
        "--seed",
        str(int(args.seed)),
        "--num-workers",
        str(int(args.num_workers)),
        "--hidden",
        *[str(int(v)) for v in args.hidden],
        "--pin-memory" if bool(args.pin_memory) else "--no-pin-memory",
    ]
    return command


def run_task(args: argparse.Namespace, task_name: str, task_root: Path, repeat_index: int) -> Dict[str, Any]:
    task_root.mkdir(parents=True, exist_ok=True)
    if task_completed(task_root):
        return {
            "task": task_name,
            "repeat_index": int(repeat_index),
            "status": "completed",
            "task_root": str(task_root),
            "skipped": True,
        }

    cmd = build_worker_command(args, task_name, task_root)
    proc = subprocess.Popen(cmd)
    try:
        code = proc.wait()
    finally:
        terminate_child_process(proc)
    if code != 0:
        return {
            "task": task_name,
            "repeat_index": int(repeat_index),
            "status": "failed",
            "task_root": str(task_root),
            "exit_code": int(code),
            "command": cmd,
        }
    return {
        "task": task_name,
        "repeat_index": int(repeat_index),
        "status": "completed",
        "task_root": str(task_root),
        "command": cmd,
    }


def main() -> None:
    args = parse_args()
    tasks = [str(t).lower() for t in args.tasks]
    if "all" in tasks:
        tasks = list(DEFAULT_TASKS)
    else:
        allowed = set(task_names()) | set(DEFAULT_TASKS)
        tasks = [t for t in tasks if t in allowed]
    if not tasks:
        raise SystemExit("No tasks requested.")

    run_root = Path(args.run_root) if args.run_root else Path(args.results_dir) / "adp" / "w2d" / "repeat4_v1"
    run_root.mkdir(parents=True, exist_ok=True)
    resume_logs = (run_root / "training_log.txt").exists() or (run_root / "training_stats.csv").exists()
    logger = ContinuousLogger(run_root, "adp_w2d_suite", "adp_width_to_depth", resume=resume_logs)
    logger.log_console(f"Run root: {run_root}")
    logger.log_console(f"Tasks: {tasks}")
    logger.log_console(f"Batch size: {int(args.batch_size)}")
    logger.log_console(f"Hidden seed: {list(map(int, args.hidden))}")
    logger.log_console(
        "Repeat counts: "
        + ", ".join(f"{task}={repeat_count_for_task(task, int(args.repeat_count))}" for task in tasks)
    )
    logger.log_console(f"Concurrency: {int(args.concurrency)}")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    pending: Deque[Tuple[int, str, Path]] = deque()
    reports: List[Dict[str, Any]] = []

    try:
        max_repeat_count = max(repeat_count_for_task(task, int(args.repeat_count)) for task in tasks)
        for repeat_index in range(1, max_repeat_count + 1):
            repeat_root = run_root / f"repeat_{repeat_index:02d}"
            repeat_root.mkdir(parents=True, exist_ok=True)
            pending.clear()
            for task_name in tasks:
                if repeat_index <= repeat_count_for_task(task_name, int(args.repeat_count)):
                    pending.append((repeat_index, task_name, repeat_root / task_name))

            active: Dict[subprocess.Popen[Any], Tuple[int, str, Path, List[str]]] = {}
            logger.log_console(
                f"[REPEAT] start repeat={repeat_index}/{max_repeat_count} root={repeat_root}"
            )

            while pending or active:
                while pending and len(active) < int(args.concurrency):
                    current_repeat, task_name, task_root = pending.popleft()
                    if task_completed(task_root):
                        reports.append(
                            {
                                "repeat_index": int(current_repeat),
                                "task": task_name,
                                "status": "completed",
                                "task_root": str(task_root),
                                "skipped": True,
                            }
                        )
                        continue
                    cmd = build_worker_command(args, task_name, task_root)
                    task_root.mkdir(parents=True, exist_ok=True)
                    proc = subprocess.Popen(cmd)
                    active[proc] = (current_repeat, task_name, task_root, cmd)

                if not active:
                    continue

                finished: List[subprocess.Popen[Any]] = []
                for proc in list(active):
                    code = proc.poll()
                    if code is None:
                        continue
                    current_repeat, task_name, task_root, cmd = active[proc]
                    terminate_child_process(proc)
                    finished.append(proc)
                    if code != 0:
                        logger.log_console(f"[TASK] failed repeat={current_repeat} task={task_name} code={code}")
                        reports.append(
                            {
                                "repeat_index": int(current_repeat),
                                "task": task_name,
                                "status": "failed",
                                "task_root": str(task_root),
                                "exit_code": int(code),
                                "command": cmd,
                            }
                        )
                        pending.appendleft((current_repeat, task_name, task_root))
                    else:
                        logger.log_console(f"[TASK] completed repeat={current_repeat} task={task_name}")
                        reports.append(
                            {
                                "repeat_index": int(current_repeat),
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
        try:
            active
        except NameError:
            active = {}
        for proc in list(active):
            terminate_child_process(proc)
        summary = {
            "run_root": str(run_root),
            "tasks": tasks,
            "batch_size": int(args.batch_size),
            "hidden": [int(v) for v in args.hidden],
            "adp_mode": str(args.adp_mode),
            "repeat_count": int(args.repeat_count),
            "repeat_counts": {task: repeat_count_for_task(task, int(args.repeat_count)) for task in tasks},
            "reports": reports,
        }
        (run_root / "suite_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        logger.close()


if __name__ == "__main__":
    main()
