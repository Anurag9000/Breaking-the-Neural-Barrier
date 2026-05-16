from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from DAE.DNN.run_goliath import (
    build_task,
    eval_final,
    extract_hidden_widths,
    infer_model_signature_from_state_dict,
    load_checkpoint,
    make_model,
    read_json,
    refresh_task_loaders,
    write_json,
)


EPOCH_PATTERN = re.compile(r"\bepoch=(\d+)\b")


def log_line(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def candidate_state_paths(run_root: Path) -> List[Path]:
    return sorted(run_root.rglob("candidate_state.json"))


def method_state_paths(run_root: Path) -> List[Path]:
    return sorted(run_root.rglob("method_state.json"))


def task_state_paths(run_root: Path) -> List[Path]:
    return sorted(run_root.rglob("task_state.json"))


def training_log_paths(run_root: Path) -> List[Path]:
    return sorted(run_root.rglob("training_log.txt"))


def parse_latest_epoch_from_log(log_path: Path) -> Optional[int]:
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        match = EPOCH_PATTERN.search(line)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def latest_heartbeat_file(run_root: Path) -> Optional[Path]:
    files: List[Path] = []
    files.extend(candidate_state_paths(run_root))
    files.extend(method_state_paths(run_root))
    files.extend(task_state_paths(run_root))
    files.extend(training_log_paths(run_root))
    files = [path for path in files if path.exists()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def latest_incomplete_candidate(run_root: Path) -> Optional[Path]:
    latest_path: Optional[Path] = None
    latest_mtime = -1.0
    for state_path in candidate_state_paths(run_root):
        state = load_json_if_exists(state_path)
        if not state or bool(state.get("completed", False)):
            continue
        try:
            mtime = state_path.stat().st_mtime
        except Exception:
            continue
        if mtime >= latest_mtime:
            latest_mtime = mtime
            latest_path = state_path.parent
    return latest_path


def candidate_summary_payload(candidate_dir: Path, device: torch.device) -> Optional[Dict[str, Any]]:
    state = load_json_if_exists(candidate_dir / "candidate_state.json") or {}
    metadata = load_json_if_exists(candidate_dir / "metadata.json") or {}
    checkpoint_path = candidate_dir / "checkpoint_last.pt"
    if not checkpoint_path.exists():
        checkpoint_path = candidate_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        return None

    try:
        ckpt = load_checkpoint(checkpoint_path, device)
    except Exception:
        return None

    state_dict = ckpt.get("best_state") or ckpt.get("model_state")
    if state_dict is None:
        return None

    cfg = metadata.get("config") or {}
    task_name = state.get("task") or metadata.get("task") or cfg.get("task")
    if task_name is None:
        return None

    data_dir = str(cfg.get("data_dir", "./data"))
    batch_size = int(cfg.get("batch_size", 32768))
    num_workers = int(cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 0))

    task = build_task(str(task_name), data_dir, batch_size, num_workers, seed)
    refresh_task_loaders(task, batch_size)

    reconstruct = bool(state.get("reconstruct", metadata.get("reconstruct", task.task_type == "reconstruction")))
    model_meta = metadata.get("model") or {}
    if model_meta:
        model = make_model(
            int(model_meta.get("in_dim", task.in_dim)),
            model_meta.get("hidden_widths", state.get("architecture", [])),
            int(model_meta.get("out_dim", task.out_dim)),
            bool(model_meta.get("use_bn", True)),
        ).to(device)
    else:
        in_dim, hidden_widths, out_dim, use_bn = infer_model_signature_from_state_dict(state_dict)
        model = make_model(in_dim, hidden_widths, out_dim, use_bn).to(device)

    try:
        model.load_state_dict(state_dict)
    except Exception:
        return None

    try:
        test_metrics = eval_final(model, task, device, reconstruct=reconstruct)
    except Exception:
        test_metrics = {}

    architecture: Any = state.get("architecture") or model_meta.get("hidden_widths") or []
    if not architecture and hasattr(model, "hidden_widths"):
        architecture = [int(w) for w in model.hidden_widths]
    if isinstance(architecture, dict):
        architecture = architecture.get("hidden_widths", architecture)
    if isinstance(architecture, (list, tuple)):
        architecture_payload: Any = {"hidden_widths": [int(w) for w in architecture], "in_dim": task.in_dim, "out_dim": task.out_dim, "use_bn": bool(getattr(model, "use_bn", True))}
    else:
        hidden_widths = extract_hidden_widths(architecture)
        architecture_payload = {"hidden_widths": hidden_widths, "in_dim": task.in_dim, "out_dim": task.out_dim, "use_bn": bool(getattr(model, "use_bn", True))}

    best_checkpoint = candidate_dir / "checkpoint_best.pt"
    if not best_checkpoint.exists():
        best_checkpoint = checkpoint_path

    return {
        "method": candidate_dir.parent.name,
        "candidate_dir": str(candidate_dir),
        "architecture": architecture_payload,
        "best_val": float(ckpt.get("best_val", state.get("best_val", float("inf")))),
        "best_epoch": int(ckpt.get("best_epoch", state.get("best_epoch", state.get("epoch", 0)))),
        "final_epoch": int(ckpt.get("epoch", state.get("final_epoch", state.get("epoch", 0)))),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(checkpoint_path),
        "test_metrics": test_metrics,
        "reconstruct": reconstruct,
        "watchdog_forced": True,
    }


def finalize_candidate(candidate_dir: Path, device: torch.device) -> bool:
    state_path = candidate_dir / "candidate_state.json"
    state = load_json_if_exists(state_path)
    if not state or bool(state.get("completed", False)):
        return False

    summary = candidate_summary_payload(candidate_dir, device)
    if summary is not None:
        write_json(candidate_dir / "candidate_summary.json", summary)
        state.update(summary)

    state.update(
        {
            "completed": True,
            "finalized_by_watchdog": True,
        }
    )
    write_json(state_path, state)
    return True


def terminate_process(proc: subprocess.Popen[Any], grace_seconds: int) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    deadline = time.monotonic() + float(grace_seconds)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        proc.kill()
    except Exception:
        pass


def run_supervised(command: Sequence[str], run_root: Path, idle_seconds: int, max_restarts: int, burst_limit: int, burst_window_seconds: int, poll_seconds: int, grace_seconds: int) -> int:
    if not command:
        raise ValueError("No command supplied to watchdog supervisor")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restart_count = 0
    first_hiccup_at: Optional[float] = None
    hiccup_restarts = 0
    current_heartbeat_path: Optional[Path] = None
    current_heartbeat_mtime: Optional[float] = None
    current_progress_value: Optional[int] = None

    while True:
        log_line(f"Starting supervised command: {' '.join(command)}")
        proc = subprocess.Popen(list(command))
        last_progress_at = time.monotonic()

        while proc.poll() is None:
            heartbeat_path = latest_heartbeat_file(run_root)
            heartbeat_mtime = None
            progress_value = None
            if heartbeat_path is not None:
                try:
                    heartbeat_mtime = heartbeat_path.stat().st_mtime
                except Exception:
                    heartbeat_mtime = None
                if heartbeat_path.name == "candidate_state.json":
                    state = load_json_if_exists(heartbeat_path) or {}
                    progress_value = int(state.get("epoch") or state.get("final_epoch") or 0)
                elif heartbeat_path.name == "training_log.txt":
                    progress_value = parse_latest_epoch_from_log(heartbeat_path)
                elif heartbeat_path.name == "method_state.json":
                    state = load_json_if_exists(heartbeat_path) or {}
                    progress_value = int(state.get("candidate_index") or state.get("next_candidate_index") or 0)
                elif heartbeat_path.name == "task_state.json":
                    state = load_json_if_exists(heartbeat_path) or {}
                    progress_value = int(state.get("next_phase_index") or state.get("next_method_index") or 0)

            if heartbeat_path is not None:
                changed = (
                    heartbeat_path != current_heartbeat_path
                    or heartbeat_mtime != current_heartbeat_mtime
                    or progress_value != current_progress_value
                )
                if changed:
                    current_heartbeat_path = heartbeat_path
                    current_heartbeat_mtime = heartbeat_mtime
                    current_progress_value = progress_value
                    last_progress_at = time.monotonic()
                    first_hiccup_at = None
                    hiccup_restarts = 0

            idle_for = time.monotonic() - last_progress_at
            if idle_for >= float(idle_seconds):
                log_line(
                    f"Watchdog stall detected after {int(idle_for)}s without progress. Restart count={restart_count}."
                )
                terminate_process(proc, grace_seconds)
                restart_count += 1
                if first_hiccup_at is None:
                    first_hiccup_at = time.monotonic()
                    hiccup_restarts = 0
                hiccup_restarts += 1

                if restart_count >= int(max_restarts) or (
                    first_hiccup_at is not None
                    and (time.monotonic() - first_hiccup_at) <= float(burst_window_seconds)
                    and hiccup_restarts >= int(burst_limit)
                ):
                    candidate_dir = latest_incomplete_candidate(run_root)
                    if candidate_dir is not None:
                        log_line(f"Force-finalizing stuck candidate: {candidate_dir}")
                        finalize_candidate(candidate_dir, device)
                    restart_count = 0
                    first_hiccup_at = None
                    hiccup_restarts = 0

                break

            time.sleep(max(1, int(poll_seconds)))

        exit_code = proc.wait()
        if exit_code == 0:
            log_line("Supervised command completed successfully.")
            return 0

        log_line(f"Supervised command exited with code {exit_code}; restarting from saved state.")
        if restart_count >= int(max_restarts):
            candidate_dir = latest_incomplete_candidate(run_root)
            if candidate_dir is not None:
                log_line(f"Final restart budget exhausted; force-finalizing {candidate_dir}.")
                finalize_candidate(candidate_dir, device)
            restart_count = 0
            first_hiccup_at = None
            hiccup_restarts = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watchdog supervisor for resumable DAE/DNN training runs")
    p.add_argument("--run-root", type=str, required=True, help="Run root to monitor")
    p.add_argument("--idle-seconds", type=int, default=120, help="Kill the child if no progress is observed for this long")
    p.add_argument("--max-restarts", type=int, default=5, help="Maximum restarts before force-finalizing the stuck unit")
    p.add_argument("--burst-limit", type=int, default=3, help="Restart limit inside the burst window before force-finalizing the stuck unit")
    p.add_argument("--burst-window-seconds", type=int, default=600, help="Burst window for repeated stalls")
    p.add_argument("--poll-seconds", type=int, default=10, help="Polling interval while supervising")
    p.add_argument("--grace-seconds", type=int, default=20, help="Time to wait for a graceful shutdown before SIGKILL")
    p.add_argument("command", nargs=argparse.REMAINDER, help="Command to supervise; use `--` before the command")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No command provided. Use `--` followed by the training command to supervise.")
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    exit_code = run_supervised(
        command=command,
        run_root=run_root,
        idle_seconds=int(args.idle_seconds),
        max_restarts=int(args.max_restarts),
        burst_limit=int(args.burst_limit),
        burst_window_seconds=int(args.burst_window_seconds),
        poll_seconds=int(args.poll_seconds),
        grace_seconds=int(args.grace_seconds),
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
