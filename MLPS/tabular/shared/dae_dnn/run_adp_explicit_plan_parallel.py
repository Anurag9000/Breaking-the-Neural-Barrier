from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

from MLPS.tabular.shared.dae_dnn.runtime_tuning import bootstrap_runtime, launcher_child_env
from utils.adp_logging import ContinuousLogger

try:  # pragma: no cover - import shim for direct script execution
    import run_goliath as rg
except ModuleNotFoundError:  # pragma: no cover - import shim for package-style imports
    from MLPS.tabular.shared.dae_dnn import run_goliath as rg

from MLPS.tabular.shared.dae_dnn.run_adp_w2d_suite_parallel import (
    build_worker_command,
    task_completed,
    terminate_child_process,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel ADP width-to-depth launcher for an explicit task/repeat plan.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", required=True)
    p.add_argument("--plan-file", required=True, help="JSON file listing phases and jobs to run.")
    p.add_argument("--batch-size", type=int, default=186240)
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
    p.add_argument("--concurrency", type=int, default=7)
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=False)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    return p.parse_args()


def load_plan(plan_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    phases = payload.get("phases")
    if not isinstance(phases, list) or not phases:
        raise SystemExit(f"Invalid plan file: {plan_path}")
    return phases


def main() -> None:
    args = parse_args()
    os.environ["TABULAR_CPU_JOB_CONCURRENCY"] = str(int(args.concurrency))
    bootstrap_runtime("run_adp_explicit_plan_parallel")
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.plan_file)
    phases = load_plan(plan_path)

    resume_logs = (run_root / "training_log.txt").exists() or (run_root / "training_stats.csv").exists()
    logger = ContinuousLogger(run_root, "adp_w2d_explicit_plan", "adp_width_to_depth", resume=resume_logs)
    logger.log_console(f"Run root: {run_root}")
    logger.log_console(f"Plan file: {plan_path}")
    logger.log_console(f"Batch size: {int(args.batch_size)}")
    logger.log_console(f"Hidden seed: {list(map(int, args.hidden))}")
    logger.log_console(f"Concurrency: {int(args.concurrency)}")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    reports: List[Dict[str, Any]] = []
    active: Dict[subprocess.Popen[Any], Tuple[str, int, str, Path, List[str]]] = {}

    try:
        for phase_index, phase in enumerate(phases, start=1):
            phase_name = str(phase.get("name", f"phase_{phase_index:02d}"))
            jobs = phase.get("jobs", [])
            if not isinstance(jobs, list):
                raise SystemExit(f"Invalid jobs in phase {phase_name}")

            pending: Deque[Tuple[int, str, Path]] = deque()
            tasks_by_repeat: Dict[int, List[str]] = defaultdict(list)
            for job in jobs:
                repeat_index = int(job["repeat"])
                task_name = str(job["task"]).lower()
                task_root = run_root / f"repeat_{repeat_index:02d}" / task_name
                pending.append((repeat_index, task_name, task_root))
                tasks_by_repeat[repeat_index].append(task_name)

            repeat_summary = ", ".join(
                f"repeat_{repeat_index:02d}={tasks}"
                for repeat_index, tasks in sorted(tasks_by_repeat.items())
            )
            logger.log_console(f"[PHASE] start name={phase_name} jobs={len(jobs)} {repeat_summary}")

            while pending or active:
                while pending and len(active) < int(args.concurrency):
                    current_repeat, task_name, task_root = pending.popleft()
                    if task_completed(task_root):
                        reports.append(
                            {
                                "phase": phase_name,
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
                    proc = subprocess.Popen(cmd, env=launcher_child_env(concurrency_hint=len(active) + 1))
                    active[proc] = (phase_name, current_repeat, task_name, task_root, cmd)

                if not active:
                    continue

                finished: List[subprocess.Popen[Any]] = []
                for proc in list(active):
                    code = proc.poll()
                    if code is None:
                        continue
                    current_phase, current_repeat, task_name, task_root, cmd = active[proc]
                    terminate_child_process(proc)
                    finished.append(proc)
                    if code != 0:
                        logger.log_console(
                            f"[TASK] failed phase={current_phase} repeat={current_repeat} task={task_name} code={code}"
                        )
                        reports.append(
                            {
                                "phase": current_phase,
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
                        logger.log_console(
                            f"[TASK] completed phase={current_phase} repeat={current_repeat} task={task_name}"
                        )
                        reports.append(
                            {
                                "phase": current_phase,
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
        for proc in list(active):
            terminate_child_process(proc)
        summary = {
            "run_root": str(run_root),
            "plan_file": str(plan_path),
            "batch_size": int(args.batch_size),
            "hidden": [int(v) for v in args.hidden],
            "adp_mode": str(args.adp_mode),
            "phases": phases,
            "reports": reports,
        }
        (run_root / "suite_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        logger.close()


if __name__ == "__main__":
    main()
