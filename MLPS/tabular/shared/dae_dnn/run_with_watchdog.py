from __future__ import annotations

import argparse
import gc
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from MLPS.tabular.shared.dae_dnn.run_goliath import (
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


@dataclass(frozen=True)
class HeartbeatSnapshot:
    latest_path: Optional[str]
    latest_mtime_ns: int
    epoch_events: int
    candidate_epoch_total: int
    method_progress_total: int
    task_progress_total: int


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


def sample_host_available_mib() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(int(parts[1]) // 1024)
                    break
    except Exception:
        return None
    return None


def sample_host_total_mib() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(int(parts[1]) // 1024)
                    break
    except Exception:
        return None
    return None


def sample_host_used_mib() -> Optional[int]:
    total = sample_host_total_mib()
    available = sample_host_available_mib()
    if total is None or available is None:
        return None
    return max(0, int(total) - int(available))


def sample_gpu_free_mib(device_index: int = 0) -> Optional[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={int(device_index)}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    text = out.decode("utf-8").strip().splitlines()
    if not text:
        return None
    try:
        return int(float(text[0].strip()))
    except Exception:
        return None


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


def count_epoch_events_in_log(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0
    return len(EPOCH_PATTERN.findall(text))


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


def heartbeat_snapshot(run_root: Path) -> HeartbeatSnapshot:
    candidate_paths = [path for path in candidate_state_paths(run_root) if path.exists()]
    method_paths = [path for path in method_state_paths(run_root) if path.exists()]
    task_paths = [path for path in task_state_paths(run_root) if path.exists()]
    log_paths = [path for path in training_log_paths(run_root) if path.exists()]

    files: List[Path] = []
    files.extend(candidate_paths)
    files.extend(method_paths)
    files.extend(task_paths)
    files.extend(log_paths)

    latest_path: Optional[Path] = None
    latest_mtime_ns = 0
    for path in files:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except Exception:
            continue
        if latest_path is None or mtime_ns >= latest_mtime_ns:
            latest_path = path
            latest_mtime_ns = mtime_ns

    epoch_events = sum(count_epoch_events_in_log(path) for path in log_paths)
    candidate_epoch_total = 0
    for path in candidate_paths:
        state = load_json_if_exists(path) or {}
        candidate_epoch_total += int(state.get("epoch") or state.get("final_epoch") or 0)

    method_progress_total = 0
    for path in method_paths:
        state = load_json_if_exists(path) or {}
        method_progress_total += int(state.get("candidate_index") or state.get("next_candidate_index") or 0)

    task_progress_total = 0
    for path in task_paths:
        state = load_json_if_exists(path) or {}
        task_progress_total += int(state.get("next_phase_index") or state.get("next_method_index") or 0)

    return HeartbeatSnapshot(
        latest_path=str(latest_path) if latest_path is not None else None,
        latest_mtime_ns=latest_mtime_ns,
        epoch_events=epoch_events,
        candidate_epoch_total=candidate_epoch_total,
        method_progress_total=method_progress_total,
        task_progress_total=task_progress_total,
    )


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
    batch_size = max(1, int(cfg.get("batch_size", 32768) or 1))
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


def finalize_candidate(candidate_dir: Path, device: torch.device, *, materialize_summary: bool = True, failed: bool = False) -> bool:
    state_path = candidate_dir / "candidate_state.json"
    state = load_json_if_exists(state_path)
    if not state or bool(state.get("completed", False)):
        return False

    if materialize_summary:
        summary = candidate_summary_payload(candidate_dir, device)
        if summary is not None:
            write_json(candidate_dir / "candidate_summary.json", summary)
            state.update(summary)

    state.update(
        {
            "completed": True,
            "finalized_by_watchdog": True,
            "failed": bool(failed or state.get("failed", False)),
        }
    )
    write_json(state_path, state)
    return True


def terminate_process(proc: subprocess.Popen[Any], grace_seconds: int) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
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
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def flush_local_cuda() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def ensure_run_root_arg(command: Sequence[str], run_root: Path) -> List[str]:
    prepared = list(command)
    if "--run-root" in prepared:
        return prepared
    for token in prepared:
        if token.endswith("run_goliath.py") or token.endswith("run_search_suite.py"):
            return prepared + ["--run-root", str(run_root)]
    return prepared


def register_hiccup(
    *,
    restart_count: int,
    first_hiccup_at: Optional[float],
    hiccup_restarts: int,
    now: float,
) -> tuple[int, float, int]:
    restart_count += 1
    if first_hiccup_at is None:
        first_hiccup_at = now
        hiccup_restarts = 0
    hiccup_restarts += 1
    return restart_count, first_hiccup_at, hiccup_restarts


def run_supervised(
    command: Sequence[str],
    run_root: Path,
    idle_seconds: int,
    max_restarts: int,
    burst_limit: int,
    burst_window_seconds: int,
    poll_seconds: int,
    resource_poll_seconds: float,
    grace_seconds: int,
    min_host_ram_mib: int,
    max_host_ram_used_mib: int,
    min_vram_free_mib: int,
) -> int:
    if not command:
        raise ValueError("No command supplied to watchdog supervisor")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared_command = ensure_run_root_arg(command, run_root)
    restart_count = 0
    first_hiccup_at: Optional[float] = None
    hiccup_restarts = 0
    current_snapshot = HeartbeatSnapshot(
        latest_path=None,
        latest_mtime_ns=0,
        epoch_events=0,
        candidate_epoch_total=0,
        method_progress_total=0,
        task_progress_total=0,
    )
    last_resource_poll_at = 0.0

    while True:
        log_line(f"Starting supervised command: {' '.join(prepared_command)}")
        proc = subprocess.Popen(prepared_command, start_new_session=True)
        last_progress_at = time.monotonic()
        last_resource_poll_at = 0.0

        while proc.poll() is None:
            now = time.monotonic()
            snapshot = heartbeat_snapshot(run_root)
            if snapshot != current_snapshot:
                current_snapshot = snapshot
                last_progress_at = now
                first_hiccup_at = None
                hiccup_restarts = 0

            if now - last_resource_poll_at >= float(max(0.1, resource_poll_seconds)):
                host_available = sample_host_available_mib()
                host_used = sample_host_used_mib()
                gpu_free = sample_gpu_free_mib(int(torch.cuda.current_device()) if torch.cuda.is_available() else 0)
                resource_pressure = False
                pressure_bits: List[str] = []
                if int(min_host_ram_mib) > 0 and host_available is not None and host_available < int(min_host_ram_mib):
                    resource_pressure = True
                    pressure_bits.append(f"host_ram_available_mib={host_available} < {int(min_host_ram_mib)}")
                if int(max_host_ram_used_mib) > 0 and host_used is not None and host_used > int(max_host_ram_used_mib):
                    resource_pressure = True
                    pressure_bits.append(f"host_ram_used_mib={host_used} > {int(max_host_ram_used_mib)}")
                if int(min_vram_free_mib) > 0 and gpu_free is not None and gpu_free < int(min_vram_free_mib):
                    resource_pressure = True
                    pressure_bits.append(f"gpu_free_mib={gpu_free} < {int(min_vram_free_mib)}")
                if resource_pressure:
                    log_line(
                        "Resource pressure detected; terminating current process group and advancing after finalization. "
                        + "; ".join(pressure_bits)
                    )
                    terminate_process(proc, grace_seconds)
                    flush_local_cuda()
                    restart_count, first_hiccup_at, hiccup_restarts = register_hiccup(
                        restart_count=restart_count,
                        first_hiccup_at=first_hiccup_at,
                        hiccup_restarts=hiccup_restarts,
                        now=time.monotonic(),
                    )
                    candidate_dir = latest_incomplete_candidate(run_root)
                    if candidate_dir is not None:
                        log_line(f"Force-finalizing resource-pressured candidate: {candidate_dir}")
                        finalize_candidate(candidate_dir, device, materialize_summary=False, failed=True)
                    break
                last_resource_poll_at = now

            idle_for = now - last_progress_at
            if idle_for >= float(idle_seconds):
                log_line(
                    f"Watchdog stall detected after {int(idle_for)}s without progress. Restart count={restart_count}. "
                    f"latest_path={current_snapshot.latest_path} epoch_events={current_snapshot.epoch_events} "
                    f"candidate_epoch_total={current_snapshot.candidate_epoch_total} "
                    f"method_progress_total={current_snapshot.method_progress_total} "
                    f"task_progress_total={current_snapshot.task_progress_total}"
                )
                terminate_process(proc, grace_seconds)
                flush_local_cuda()
                restart_count, first_hiccup_at, hiccup_restarts = register_hiccup(
                    restart_count=restart_count,
                    first_hiccup_at=first_hiccup_at,
                    hiccup_restarts=hiccup_restarts,
                    now=time.monotonic(),
                )

                if restart_count >= int(max_restarts) or (
                    first_hiccup_at is not None
                    and (time.monotonic() - first_hiccup_at) <= float(burst_window_seconds)
                    and hiccup_restarts >= int(burst_limit)
                ):
                    candidate_dir = latest_incomplete_candidate(run_root)
                    if candidate_dir is not None:
                        log_line(f"Force-finalizing stuck candidate: {candidate_dir}")
                        finalize_candidate(candidate_dir, device, materialize_summary=True, failed=True)
                    restart_count = 0
                    first_hiccup_at = None
                    hiccup_restarts = 0

                break

            time.sleep(min(max(0.1, float(resource_poll_seconds)), max(1.0, float(poll_seconds))))

        exit_code = proc.wait()
        if exit_code == 0:
            log_line("Supervised command completed successfully.")
            return 0

        log_line(f"Supervised command exited with code {exit_code}; restarting from saved state.")
        flush_local_cuda()
        restart_count, first_hiccup_at, hiccup_restarts = register_hiccup(
            restart_count=restart_count,
            first_hiccup_at=first_hiccup_at,
            hiccup_restarts=hiccup_restarts,
            now=time.monotonic(),
        )
        if restart_count >= int(max_restarts):
            candidate_dir = latest_incomplete_candidate(run_root)
            if candidate_dir is not None:
                log_line(f"Final restart budget exhausted; force-finalizing {candidate_dir}.")
                finalize_candidate(candidate_dir, device, materialize_summary=True, failed=True)
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
    p.add_argument("--resource-poll-seconds", type=float, default=0.25, help="Polling interval for RAM/VRAM pressure checks.")
    p.add_argument("--grace-seconds", type=int, default=20, help="Time to wait for a graceful shutdown before SIGKILL")
    p.add_argument("--min-host-ram-mib", type=int, default=1024, help="Force-stop the current process group if MemAvailable drops below this value.")
    p.add_argument("--max-host-ram-used-mib", type=int, default=0, help="Force-stop the current process group if total host RAM used exceeds this value. Set 0 to disable.")
    p.add_argument("--min-vram-free-mib", type=int, default=100, help="Force-stop the current process group if free GPU VRAM drops below this value.")
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
        resource_poll_seconds=float(args.resource_poll_seconds),
        grace_seconds=int(args.grace_seconds),
        min_host_ram_mib=int(args.min_host_ram_mib),
        max_host_ram_used_mib=int(args.max_host_ram_used_mib),
        min_vram_free_mib=int(args.min_vram_free_mib),
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
