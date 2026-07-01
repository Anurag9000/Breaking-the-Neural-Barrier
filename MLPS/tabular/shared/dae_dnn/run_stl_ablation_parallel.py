from __future__ import annotations

import argparse
import math
import json
import os
import psutil
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MLPS.tabular.shared.dae_dnn.platform_runtime import (
    popen_process_group_kwargs,
    sample_host_memory_mib,
    terminate_process_tree,
)
from MLPS.tabular.shared.dae_dnn.runtime_tuning import bootstrap_runtime, detect_cpu_cores, launcher_child_env
from MLPS.tabular.shared.dae_dnn.tasks import build_task
from utils.adp_logging import ContinuousLogger

try:  # pragma: no cover - import shim for direct script execution
    import run_goliath as rg
    import run_stl_ablation as stl
except ModuleNotFoundError:  # pragma: no cover - import shim for package-style imports
    from MLPS.tabular.shared.dae_dnn import run_goliath as rg
    from MLPS.tabular.shared.dae_dnn import run_stl_ablation as stl


LAUNCHER_MANIFEST_SIGNATURE_VERSION = 3
LAUNCHER_CODE_VERSION = 3
LAUNCHER_BATCH_BACKOFF_VERSION = 1


@dataclass(frozen=True)
class ChildJob:
    task_name: str
    architecture: Tuple[int, ...]
    child_root: Path
    parameter_count: int
    depth: int
    phase_name: str


@dataclass
class ActiveChildJob:
    job: ChildJob
    cmd: List[str]
    device_mode: str
    log_path: Path
    slot_index: int
    log_handle: Optional[IO[str]] = None
    pause_requested: bool = False
    pause_reason: Optional[str] = None
    launch_count: int = 0


@dataclass(frozen=True)
class MemoryPressureSample:
    total_mib: int
    available_mib: int
    used_pct: float


@dataclass(frozen=True)
class GpuPressureSample:
    total_mib: int
    used_mib: int
    used_pct: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel STL ablation launcher with resumable child runs.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default=None)
    p.add_argument("--source-run-root", default="MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current")
    p.add_argument("--tasks", nargs="+", default=list(stl.DEFAULT_TASKS))
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-width", type=int, default=1024)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--width-stage-margin-patience", type=int, default=10)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--min-width", type=int, default=1)
    p.add_argument("--width-step", type=int, default=1)
    p.add_argument("--width-count-per-depth", type=int, default=10)
    p.add_argument("--min-depth", type=int, default=1)
    p.add_argument(
        "--param-band",
        nargs=2,
        type=int,
        metavar=("PARAM_EXP_START", "PARAM_EXP_END"),
        default=None,
        help="Split the massive STL run by parameter-count decades, e.g. 1 3 for 10^1-10^3.",
    )
    p.add_argument("--repeat-count", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument(
        "--job-start-index",
        type=int,
        default=0,
        help="Zero-based start index into the globally sorted job queue. Useful for splitting a band across machines.",
    )
    p.add_argument(
        "--job-limit",
        type=int,
        default=0,
        help="Maximum number of jobs to take from the globally sorted queue after --job-start-index. 0 means no limit.",
    )
    p.add_argument(
        "--concurrency-file",
        default=None,
        help="Optional text file containing the concurrency value to use instead of --concurrency.",
    )
    p.add_argument(
        "--scheduler",
        choices=["pressure_aware", "gpu_first", "fixed"],
        default="pressure_aware",
        help=(
            "Scheduler mode: 'pressure_aware' (single RAM+GPU gate), "
            "'gpu_first' (dual-gate: GPU gate reopens only on GPU job completion or "
            ">500 MiB GPU VRAM drop; CPU gate uses existing RAM logic), "
            "or 'fixed' (legacy fixed-slot)."
        ),
    )
    p.add_argument(
        "--max-active-jobs",
        type=int,
        default=0,
        help="Hard cap for pressure-aware child jobs. 0 means no hard child-count cap beyond pressure gating.",
    )
    p.add_argument(
        "--host-ram-pressure-limit-pct",
        type=float,
        default=85.0,
        help="Pause the largest active child when used host RAM exceeds this percentage.",
    )
    p.add_argument(
        "--host-ram-resume-pct",
        type=float,
        default=80.0,
        help="Only launch or relaunch a child when used host RAM is at or below this percentage.",
    )
    p.add_argument(
        "--pressure-poll-interval-sec",
        type=float,
        default=0.5,
        help="Polling interval for child completion and host RAM pressure checks.",
    )
    p.add_argument(
        "--post-launch-sample-delay-sec",
        type=float,
        default=30.0,
        help="Delay after each child launch before the next pressure sample and admission decision.",
    )
    p.add_argument(
        "--batch-backoff-factor",
        type=float,
        default=0.5,
        help="Multiply the effective batch size by this factor after a pressure stall with no active children remaining.",
    )
    p.add_argument(
        "--max-retries-per-job",
        type=int,
        default=0,
        help="Legacy compatibility flag. Pressure-aware mode now requeues failed children indefinitely; 0 reflects unlimited retries.",
    )
    p.add_argument(
        "--gpu-memory-pressure-limit-pct",
        type=float,
        default=90.0,
        help="Pause the largest active GPU child when used GPU memory exceeds this percentage.",
    )
    p.add_argument(
        "--gpu-memory-resume-pct",
        type=float,
        default=85.0,
        help="Only launch or relaunch a child on GPU when used GPU memory is at or below this percentage.",
    )
    p.add_argument(
        "--gpu-device-index",
        type=int,
        default=0,
        help="CUDA device index to use for GPU child launches.",
    )
    p.add_argument(
        "--max-active-gpu-jobs",
        type=int,
        default=0,
        help="Maximum concurrent GPU children. 0 means memory-pressure driven.",
    )
    p.add_argument("--stl-width", type=int, default=128)
    p.add_argument("--stl-depth", type=int, default=2)
    p.add_argument("--metrics-every", type=int, default=0)
    p.add_argument(
        "--legacy-architecture-grid",
        action="store_true",
        default=False,
        help="Use the old fixed width x depth sweep instead of parameter-matched depth families.",
    )
    p.add_argument("--use-bn", action="store_true", default=True)
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    return p.parse_args()


def resolve_concurrency(args: argparse.Namespace) -> int:
    concurrency = int(args.concurrency)
    concurrency_file = getattr(args, "concurrency_file", None)
    if concurrency_file:
        path = Path(str(concurrency_file))
        if not path.exists():
            raise SystemExit(f"Missing concurrency file: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit(f"Empty concurrency file: {path}")
        try:
            concurrency = int(text.splitlines()[0].strip())
        except Exception as exc:
            raise SystemExit(f"Invalid concurrency file contents in {path}: {text!r}") from exc
    return max(1, int(concurrency))


def build_worker_command(
    *,
    args: argparse.Namespace,
    task_name: str,
    architecture: Sequence[int],
    child_run_root: Path,
    device_mode: str,
    batch_size: int,
) -> List[str]:
    command = [
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
        "--repeat-count",
        str(int(args.repeat_count)),
        "--batch-size",
        str(int(batch_size)),
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
        "--width-count-per-depth",
        str(int(args.width_count_per_depth)),
        "--min-depth",
        str(int(args.min_depth)),
        "--stl-width",
        str(int(args.stl_width)),
        "--stl-depth",
        str(int(args.stl_depth)),
        "--metrics-every",
        str(int(args.metrics_every)),
        "--device",
        str(device_mode),
    ]
    if not bool(args.use_bn):
        command.append("--no-bn")
    if bool(getattr(args, "legacy_architecture_grid", False)):
        command.append("--legacy-architecture-grid")
    if getattr(args, "param_band", None):
        command.extend(
            [
                "--param-band",
                str(int(args.param_band[0])),
                str(int(args.param_band[1])),
            ]
        )
    return command


def batch_backoff_state_path(run_root: Path) -> Path:
    return run_root / "batch_backoff_state.json"


def load_batch_backoff_state(run_root: Path) -> Dict[str, Any]:
    data = rg.load_json_if_exists(batch_backoff_state_path(run_root))
    if not isinstance(data, dict):
        return {
            "version": LAUNCHER_BATCH_BACKOFF_VERSION,
            "batch_scale": 1.0,
            "backoff_count": 0,
            "pressure_backoff_pending": False,
            "last_backoff_reason": None,
            "last_backoff_at": None,
        }
    state = {
        "version": int(data.get("version", LAUNCHER_BATCH_BACKOFF_VERSION)),
        "batch_scale": float(data.get("batch_scale", 1.0) or 1.0),
        "backoff_count": int(data.get("backoff_count", 0) or 0),
        "pressure_backoff_pending": bool(data.get("pressure_backoff_pending", False)),
        "last_backoff_reason": data.get("last_backoff_reason"),
        "last_backoff_at": data.get("last_backoff_at"),
    }
    if state["batch_scale"] <= 0.0:
        state["batch_scale"] = 1.0
    return state


def write_batch_backoff_state(run_root: Path, state: Dict[str, Any]) -> None:
    payload = {
        "version": LAUNCHER_BATCH_BACKOFF_VERSION,
        "batch_scale": float(state.get("batch_scale", 1.0)),
        "backoff_count": int(state.get("backoff_count", 0)),
        "pressure_backoff_pending": bool(state.get("pressure_backoff_pending", False)),
        "last_backoff_reason": state.get("last_backoff_reason"),
        "last_backoff_at": state.get("last_backoff_at"),
    }
    rg.write_json(batch_backoff_state_path(run_root), payload)


def scale_batch_size(batch_size: int, batch_scale: float) -> int:
    base = max(1, int(batch_size))
    scale = max(0.0, float(batch_scale))
    return max(1, int(math.floor(float(base) * scale)))


def resolve_task_base_batch_sizes(args: argparse.Namespace, tasks: Sequence[str]) -> Dict[str, int]:
    base_batch_sizes: Dict[str, int] = {}
    for task_name in tasks:
        try:
            task = build_task(task_name, args.data_dir, 1, 0, int(args.seed), pin_memory=False)
            base_batch_sizes[task_name] = int(stl.stl_batch_size_for_task(task_name, task, int(args.batch_size)))
        except Exception:
            base_batch_sizes[task_name] = max(1, int(args.batch_size) if int(args.batch_size) > 0 else 1)
    return base_batch_sizes


def apply_batch_backoff(run_root: Path, state: Dict[str, Any], factor: float, reason: str) -> Dict[str, Any]:
    prev_scale = float(state.get("batch_scale", 1.0) or 1.0)
    factor = float(factor)
    if factor <= 0.0 or factor >= 1.0:
        raise ValueError(f"batch backoff factor must be between 0 and 1, got {factor!r}")
    new_scale = max(1.0e-6, prev_scale * factor)
    state = dict(state)
    state["batch_scale"] = new_scale
    state["backoff_count"] = int(state.get("backoff_count", 0)) + 1
    state["pressure_backoff_pending"] = False
    state["last_backoff_reason"] = str(reason)
    state["last_backoff_at"] = time.time()
    write_batch_backoff_state(run_root, state)
    return state


def child_summary_path(child_run_root: Path, task_name: str) -> Path:
    return child_run_root / task_name / "ablation_summary.json"


def child_state_path(child_run_root: Path) -> Path:
    return child_run_root / "child_run_state.json"


def child_process_log_path(child_run_root: Path) -> Path:
    return child_run_root / "_child_process.log"


def load_child_state(child_run_root: Path) -> Dict[str, Any]:
    data = rg.load_json_if_exists(child_state_path(child_run_root))
    return data if isinstance(data, dict) else {}


def write_child_state(child_run_root: Path, payload: Dict[str, Any]) -> None:
    rg.write_json(child_state_path(child_run_root), payload)


def update_child_state(child_run_root: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    state = load_child_state(child_run_root)
    state.update(updates)
    write_child_state(child_run_root, state)
    return state


def child_completed(child_run_root: Path, task_name: str) -> bool:
    task_state = rg.load_json_if_exists(child_run_root / task_name / "ablation_state.json") or {}
    if bool(task_state.get("failed", False)):
        return False
    if bool(task_state.get("completed", False)):
        return True

    state = load_child_state(child_run_root)
    if bool(state.get("failed", False)):
        return False
    if bool(state.get("completed", False)):
        return True
    summary_path = child_summary_path(child_run_root, task_name)
    plot_path = child_run_root / task_name / "ablation_loss_vs_params.png"
    data = rg.load_json_if_exists(summary_path) or {}
    return bool(data.get("ablation_stl_runs")) and plot_path.exists()


def resolve_child_root(child_base: Path, phase_name: str) -> Path:
    candidates = [
        child_base / phase_name,
        *sorted((p for p in child_base.glob(f"**/{phase_name}") if p.is_dir()), key=lambda p: (len(p.parts), str(p))),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return child_base / phase_name


def load_task_child_summary(child_run_root: Path, task_name: str) -> Dict[str, Any]:
    path = child_summary_path(child_run_root, task_name)
    data = rg.load_json_if_exists(path)
    if not isinstance(data, dict):
        state = load_child_state(child_run_root)
        if bool(state.get("failed", False)):
            return {
                "task": task_name,
                "source_adp_runs": [],
                "source_paired_stl_runs": [],
                "ablation_stl_runs": [],
                "comparisons": [],
                "best_ablation": None,
                "failed": True,
                "failed_architecture": state.get("architecture"),
                "failed_exit_code": state.get("exit_code"),
                "failed_command": state.get("command"),
            }
        raise FileNotFoundError(f"Missing child summary: {path}")
    return data


def mark_child_running(child_root: Path, job: ChildJob, cmd: Sequence[str], launch_count: int, batch_size: int) -> None:
    existing = load_child_state(child_root)
    update_child_state(
        child_root,
        {
            "task": job.task_name,
            "architecture": [int(v) for v in job.architecture],
            "phase_name": job.phase_name,
            "parameter_count": int(job.parameter_count),
            "command": list(cmd),
            "launch_count": int(launch_count),
            "batch_size": int(batch_size),
            "pause_count": int(existing.get("pause_count", 0)),
            "failure_count": int(existing.get("failure_count", 0)),
            "completed": False,
            "failed": False,
            "status": "running",
            "last_started_at": time.time(),
        },
    )


def mark_child_pause_requested(child_root: Path, job: ChildJob, reason: str) -> None:
    existing = load_child_state(child_root)
    update_child_state(
        child_root,
        {
            "task": job.task_name,
            "architecture": [int(v) for v in job.architecture],
            "phase_name": job.phase_name,
            "parameter_count": int(job.parameter_count),
            "completed": False,
            "failed": False,
            "status": "pausing",
            "pause_requested": True,
            "pause_count": int(existing.get("pause_count", 0)),
            "last_pause_reason": str(reason),
            "last_pause_requested_at": time.time(),
        },
    )


def mark_child_paused(child_root: Path, job: ChildJob, exit_code: int, reason: str) -> None:
    existing = load_child_state(child_root)
    update_child_state(
        child_root,
        {
            "task": job.task_name,
            "architecture": [int(v) for v in job.architecture],
            "phase_name": job.phase_name,
            "parameter_count": int(job.parameter_count),
            "completed": False,
            "failed": False,
            "status": "paused",
            "pause_requested": False,
            "pause_count": int(existing.get("pause_count", 0)) + 1,
            "last_pause_reason": str(reason),
            "last_exit_code": int(exit_code),
            "last_paused_at": time.time(),
        },
    )


def mark_child_retrying(child_root: Path, job: ChildJob, exit_code: int, cmd: Sequence[str], failure_count: int) -> None:
    existing = load_child_state(child_root)
    update_child_state(
        child_root,
        {
            "task": job.task_name,
            "architecture": [int(v) for v in job.architecture],
            "phase_name": job.phase_name,
            "parameter_count": int(job.parameter_count),
            "command": list(cmd),
            "completed": False,
            "failed": False,
            "status": "retrying",
            "pause_requested": False,
            "pause_count": int(existing.get("pause_count", 0)),
            "failure_count": int(failure_count),
            "last_exit_code": int(exit_code),
            "last_failed_at": time.time(),
        },
    )


def mark_child_completed(child_root: Path, job: ChildJob) -> None:
    existing = load_child_state(child_root)
    update_child_state(
        child_root,
        {
            "task": job.task_name,
            "architecture": [int(v) for v in job.architecture],
            "phase_name": job.phase_name,
            "parameter_count": int(job.parameter_count),
            "completed": True,
            "failed": False,
            "status": "completed",
            "pause_requested": False,
            "pause_count": int(existing.get("pause_count", 0)),
            "failure_count": int(existing.get("failure_count", 0)),
            "completed_at": time.time(),
        },
    )


def terminate_child_process(proc: subprocess.Popen[Any], timeout_sec: float = 10.0) -> None:
    terminate_process_tree(proc, timeout_sec=timeout_sec)


def close_child_log(handle: Optional[IO[str]]) -> None:
    if handle is None:
        return
    try:
        handle.flush()
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass


def child_log_indicates_cuda_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    markers = (
        "cuda error: out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cuda error: cublas_status_alloc_failed",
    )
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


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
                curve_rows.append(
                    {
                        "task": task_name,
                        "repeat": int(entry.get("repeat", 1)),
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


def sample_host_memory_pressure() -> MemoryPressureSample:
    total_mib, available_mib = sample_host_memory_mib()
    if total_mib <= 0:
        total_mib = 1
    if available_mib < 0:
        available_mib = 0
    used_pct = max(0.0, min(100.0, (1.0 - (float(available_mib) / float(total_mib))) * 100.0))
    return MemoryPressureSample(total_mib=int(total_mib), available_mib=int(available_mib), used_pct=float(used_pct))


def sample_gpu_memory_pressure(device_index: int = 0) -> GpuPressureSample:
    total_mib = 0
    used_mib = 0
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={int(device_index)}",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        row = out.decode("utf-8").strip().splitlines()[0]
        total_text, used_text = [part.strip() for part in row.split(",", 1)]
        total_mib = int(float(total_text))
        used_mib = int(float(used_text))
    except Exception:
        pass
    if total_mib <= 0:
        return GpuPressureSample(total_mib=0, used_mib=0, used_pct=0.0)
    used_pct = max(0.0, min(100.0, (float(used_mib) / float(total_mib)) * 100.0))
    return GpuPressureSample(total_mib=int(total_mib), used_mib=int(used_mib), used_pct=float(used_pct))

def sample_python_host_memory_mib() -> float:
    total_py_mib = 0.0
    for p in psutil.process_iter(['name', 'memory_info']):
        try:
            name = p.info['name']
            if name and ('python' in name.lower() or 'python3' in name.lower()):
                total_py_mib += p.info['memory_info'].rss / 1048576.0
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, TypeError, KeyError):
            pass
    return total_py_mib

def sample_python_gpu_memory_mib(device_index: int = 0) -> float:
    total_py_gpu_mib = 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device_index}", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        for line in out.decode("utf-8").strip().splitlines():
            parts = line.split(',')
            if len(parts) >= 2:
                pname = parts[0].strip()
                mem_str = parts[1].strip()
                if mem_str.isdigit() and ('python' in pname.lower() or 'python3' in pname.lower()):
                    total_py_gpu_mib += float(mem_str)
    except Exception:
        pass
    return total_py_gpu_mib


def concrete_job_sort_key(job: ChildJob) -> Tuple[int, int, str, Tuple[int, ...]]:
    return (int(job.parameter_count), int(job.depth), str(job.task_name), tuple(job.architecture))


def child_has_resume_state(job: ChildJob) -> bool:
    task_root = job.child_root / job.task_name
    if (task_root / "ablation_state.json").exists():
        return True
    if (task_root / "ablation_summary.json").exists():
        return True
    if child_state_path(job.child_root).exists():
        return True
    try:
        return any(job.child_root.iterdir())
    except Exception:
        return False


def pending_job_sort_key(job: ChildJob) -> Tuple[int, int, int, str, Tuple[int, ...]]:
    resume_rank = 0 if child_has_resume_state(job) and not child_completed(job.child_root, job.task_name) else 1
    return (resume_rank, int(job.parameter_count), int(job.depth), str(job.task_name), tuple(job.architecture))


def job_manifest_path(run_root: Path) -> Path:
    return run_root / "job_manifest.json"


def job_manifest_signature(args: argparse.Namespace, tasks: Sequence[str], run_root: Path) -> Dict[str, Any]:
    return {
        # Only plan-shaping inputs belong here. Scheduler/runtime knobs such as
        # concurrency, batch sizing, optimizer settings, and pressure thresholds
        # intentionally do not invalidate the cached manifest because they do
        # not change the concrete job set.
        "version": LAUNCHER_MANIFEST_SIGNATURE_VERSION,
        "launcher_code_version": LAUNCHER_CODE_VERSION,
        "tasks": sorted(str(task).lower() for task in tasks),
        "param_band": list(stl.normalize_param_band(getattr(args, "param_band", None))) if getattr(args, "param_band", None) else None,
        "repeat_count": int(args.repeat_count),
        "min_width": int(args.min_width),
        "width_step": int(args.width_step),
        "max_width": int(args.max_width),
        "min_depth": int(args.min_depth),
        "max_depth": int(args.max_depth),
        "width_count_per_depth": int(args.width_count_per_depth),
        "use_bn": bool(args.use_bn),
        "legacy_architecture_grid": bool(args.legacy_architecture_grid),
    }


def load_job_manifest(run_root: Path, signature: Dict[str, Any]) -> Optional[Dict[str, List[ChildJob]]]:
    manifest = rg.load_json_if_exists(job_manifest_path(run_root))
    if not isinstance(manifest, dict):
        return None
    if manifest.get("version") != 1 or manifest.get("signature") != signature:
        return None
    raw_jobs_by_task = manifest.get("jobs_by_task")
    if not isinstance(raw_jobs_by_task, dict):
        return None
    jobs_by_task: Dict[str, List[ChildJob]] = {}
    for task_name, raw_jobs in raw_jobs_by_task.items():
        if not isinstance(raw_jobs, list):
            return None
        task_jobs: List[ChildJob] = []
        for raw_job in raw_jobs:
            if not isinstance(raw_job, dict):
                return None
            child_root_rel = raw_job.get("child_root")
            architecture = raw_job.get("architecture")
            if not child_root_rel or not architecture:
                return None
            architecture_tuple = tuple(int(v) for v in architecture)
            task_jobs.append(
                ChildJob(
                    task_name=str(raw_job.get("task_name", task_name)),
                    architecture=architecture_tuple,
                    child_root=run_root / Path(str(child_root_rel)),
                    parameter_count=int(raw_job.get("parameter_count", 0)),
                    depth=int(raw_job.get("depth", len(architecture_tuple))),
                    phase_name=str(raw_job.get("phase_name", "")),
                )
            )
        task_jobs.sort(key=concrete_job_sort_key)
        jobs_by_task[str(task_name)] = task_jobs
    return jobs_by_task


def write_job_manifest(run_root: Path, signature: Dict[str, Any], jobs_by_task: Dict[str, List[ChildJob]]) -> None:
    serializable_jobs: Dict[str, List[Dict[str, Any]]] = {}
    for task_name, jobs in jobs_by_task.items():
        serializable_jobs[task_name] = [
            {
                "task_name": job.task_name,
                "architecture": [int(v) for v in job.architecture],
                "child_root": str(job.child_root.relative_to(run_root)),
                "parameter_count": int(job.parameter_count),
                "depth": int(job.depth),
                "phase_name": job.phase_name,
            }
            for job in jobs
        ]
    rg.write_json(job_manifest_path(run_root), {"version": 1, "signature": signature, "jobs_by_task": serializable_jobs})


def build_task_jobs(args: argparse.Namespace, tasks: Sequence[str], run_root: Path) -> Dict[str, List[ChildJob]]:
    signature = job_manifest_signature(args, tasks, run_root)
    cached = load_job_manifest(run_root, signature)
    if cached is not None:
        return cached

    cfg = stl.make_cfg(args, list(tasks), run_root)
    base_architectures = stl.build_architectures(args)
    jobs_by_task: Dict[str, List[ChildJob]] = {}
    for task_name in tasks:
        task_root = run_root / task_name
        child_base = task_root / "_children"
        child_base.mkdir(parents=True, exist_ok=True)
        task = build_task(task_name, cfg.data_dir, 1, cfg.num_workers, cfg.seed, pin_memory=False)
        task_jobs: List[ChildJob] = []
        for architecture in base_architectures:
            family = [list(architecture)]
            if cfg.parameter_matched and len(architecture) == 1:
                family = stl.parameter_matched_architectures(task, int(architecture[0]), cfg)
            family = stl.dedupe_architectures(family)
            for expanded_architecture in family:
                architecture_tuple = tuple(int(v) for v in expanded_architecture)
                phase_name = stl.phase_name_for_architecture(expanded_architecture, 1)
                child_root = resolve_child_root(child_base, phase_name)
                parameter_count = int(rg.count_model_parameters(rg.make_stl_model(task, expanded_architecture, bool(args.use_bn))))
                task_jobs.append(
                    ChildJob(
                        task_name=task_name,
                        architecture=architecture_tuple,
                        child_root=child_root,
                        parameter_count=parameter_count,
                        depth=len(architecture_tuple),
                        phase_name=phase_name,
                    )
                )
        task_jobs.sort(key=concrete_job_sort_key)
        jobs_by_task[task_name] = task_jobs
    write_job_manifest(run_root, signature, jobs_by_task)
    return jobs_by_task


def active_job_limit(args: argparse.Namespace, job_count: int) -> int:
    limit = int(getattr(args, "max_active_jobs", 0) or 0)
    if limit <= 0:
        return max(1, int(job_count))
    return max(1, int(limit))


def active_gpu_job_limit(args: argparse.Namespace) -> int:
    return max(0, int(getattr(args, "max_active_gpu_jobs", 0) or 0))


def slice_pending_jobs(jobs: Sequence[ChildJob], start_index: int, job_limit: int) -> List[ChildJob]:
    start_index = max(0, int(start_index))
    job_limit = max(0, int(job_limit))
    if start_index <= 0 and job_limit <= 0:
        return list(jobs)
    if start_index >= len(jobs):
        return []
    end_index = len(jobs) if job_limit <= 0 else min(len(jobs), start_index + job_limit)
    return list(jobs[start_index:end_index])


def launch_child_process(
    cmd: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
) -> Tuple[subprocess.Popen[Any], Optional[IO[str]]]:
    handle: Optional[IO[str]] = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        list(cmd),
        env=env,
        stdout=handle if handle is not None else None,
        stderr=handle if handle is not None else None,
        text=True,
        **popen_process_group_kwargs(),
    )
    return proc, handle


def run_parallel_task(args: argparse.Namespace, task_name: str, run_root: Path, architectures: Sequence[Sequence[int]]) -> Dict[str, Any]:
    task_root = run_root / task_name
    child_base = task_root / "_children"
    child_base.mkdir(parents=True, exist_ok=True)
    jobs: Deque[Tuple[Tuple[int, ...], Path]] = deque()
    for architecture in architectures:
        phase_name = stl.phase_name_for_architecture(architecture, 1)
        child_root = resolve_child_root(child_base, phase_name)
        jobs.append((tuple(int(v) for v in architecture), child_root))

    active: Dict[subprocess.Popen[Any], Tuple[Tuple[int, ...], Path, List[str], int]] = {}
    slot_count = max(1, min(int(args.concurrency), int(detect_cpu_cores())))
    free_slots = set(range(slot_count))
    completed_children: List[Path] = []
    launch_count = 0
    while jobs or active:
        while jobs and len(active) < int(args.concurrency) and free_slots:
            architecture, child_root = jobs.popleft()
            if child_completed(child_root, task_name):
                completed_children.append(child_root)
                continue
            cmd = build_worker_command(
                args=args,
                task_name=task_name,
                architecture=architecture,
                child_run_root=child_root,
                device_mode="auto",
                batch_size=int(args.batch_size),
            )
            child_root.mkdir(parents=True, exist_ok=True)
            slot_index = min(free_slots)
            free_slots.remove(slot_index)
            proc, _ = launch_child_process(
                cmd,
                env=launcher_child_env(
                    concurrency_hint=len(active) + 1,
                    job_key=f"{task_name}:{child_root}",
                    affinity_slot=slot_index,
                ),
            )
            active[proc] = (architecture, child_root, cmd, slot_index)
            launch_count += 1

        if not active:
            continue

        finished: List[subprocess.Popen[Any]] = []
        for proc in list(active):
            code = proc.poll()
            if code is None:
                continue
            architecture, child_root, cmd, slot_index = active[proc]
            if code != 0 and not child_completed(child_root, task_name):
                log = f"Child job failed (arch={architecture}, root={child_root}, code={code}): {' '.join(cmd)}"
                print(log, flush=True)
                terminate_child_process(proc)
                mark_child_failed(
                    child_root,
                    ChildJob(task_name=task_name, architecture=tuple(architecture), child_root=child_root, parameter_count=0, depth=len(architecture), phase_name=child_root.name),
                    code,
                    cmd,
                    1,
                )
                jobs.appendleft((architecture, child_root))
                finished.append(proc)
                continue
            terminate_child_process(proc)
            finished.append(proc)

        for proc in finished:
            architecture, child_root, _, slot_index = active.pop(proc)
            free_slots.add(slot_index)
            del architecture
            completed_children.append(child_root)

        if active:
            time.sleep(2)

    return aggregate_task(task_name, task_root, completed_children)


def run_pressure_aware(args: argparse.Namespace, run_root: Path, tasks: Sequence[str], logger: ContinuousLogger) -> List[Dict[str, Any]]:
    jobs_by_task = build_task_jobs(args, tasks, run_root)
    task_base_batch_sizes = resolve_task_base_batch_sizes(args, tasks)
    sorted_jobs = sorted((job for jobs in jobs_by_task.values() for job in jobs), key=pending_job_sort_key)
    pending_jobs = slice_pending_jobs(
        sorted_jobs,
        int(getattr(args, "job_start_index", 0) or 0),
        int(getattr(args, "job_limit", 0) or 0),
    )
    pending: Deque[ChildJob] = deque(pending_jobs)
    task_child_roots: Dict[str, List[Path]] = {task_name: [job.child_root for job in jobs] for task_name, jobs in jobs_by_task.items()}
    active: Dict[subprocess.Popen[Any], ActiveChildJob] = {}
    slot_count = max(1, int(active_job_limit(args, len(pending))))
    free_slots = set(range(slot_count))
    failure_counts: Dict[Path, int] = defaultdict(int)
    launches_total = 0
    active_limit = active_job_limit(args, len(pending))
    gpu_available = bool(torch.cuda.is_available())
    launches_enabled = True
    launch_sample_delay_sec = max(0.0, float(getattr(args, "post_launch_sample_delay_sec", 30.0)))
    launch_sample_hold_until = 0.0
    batch_backoff_state = load_batch_backoff_state(run_root)
    batch_scale = float(batch_backoff_state.get("batch_scale", 1.0))
    pressure_backoff_pending = bool(batch_backoff_state.get("pressure_backoff_pending", False))
    pressure_backoff_reason: Optional[str] = batch_backoff_state.get("last_backoff_reason")

    def build_child_env(device_mode: str) -> Dict[str, str]:
        env = os.environ.copy()
        if device_mode == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif device_mode == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_device_index))
        return env

    def choose_device_mode(
        host_pressure: MemoryPressureSample,
        gpu_pressure: GpuPressureSample,
        active_gpu_jobs: int,
    ) -> Optional[str]:
        if host_pressure.used_pct > float(args.host_ram_resume_pct):
            return None
        gpu_limit = active_gpu_job_limit(args)
        if gpu_limit > 0 and int(active_gpu_jobs) >= gpu_limit:
            return "cpu"
        if gpu_available and gpu_pressure.total_mib > 0 and gpu_pressure.used_pct <= float(args.gpu_memory_resume_pct):
            return "cuda"
        return "cpu"

    peak_paused_host_mib = 0.0
    peak_paused_gpu_mib = 0.0
    while pending or active:
        finished: List[subprocess.Popen[Any]] = []
        for proc, active_job in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            finished.append(proc)
            job = active_job.job
            close_child_log(active_job.log_handle)
            if child_completed(job.child_root, job.task_name):
                logger.log_console(
                    f"[TASK] completed task={job.task_name} phase={job.phase_name} params={job.parameter_count}"
                )
                mark_child_completed(job.child_root, job)
                pressure_backoff_pending = False
                pressure_backoff_reason = None
                batch_backoff_state.update(
                    {
                        "batch_scale": batch_scale,
                        "pressure_backoff_pending": pressure_backoff_pending,
                        "last_backoff_reason": pressure_backoff_reason,
                    }
                )
                write_batch_backoff_state(run_root, batch_backoff_state)
                if not launches_enabled:
                    logger.log_console(
                        f"[STATE] admission gate reopened by completion task={job.task_name} phase={job.phase_name} "
                        f"host_used_pct={pressure.used_pct:.2f} gpu_used_pct={gpu_pressure.used_pct:.2f}"
                    )
                launches_enabled = True
                continue
            if active_job.pause_requested:
                pause_reason = str(active_job.pause_reason or "host_ram_pressure")
                logger.log_console(
                    f"[PRESSURE] paused task={job.task_name} phase={job.phase_name} params={job.parameter_count} exit={code}"
                )
                mark_child_paused(job.child_root, job, int(code or 0), pause_reason)
                pending.appendleft(job)
                if launches_enabled:
                    launches_enabled = False
                    peak_paused_host_mib = (sample_host_memory_pressure().total_mib * sample_host_memory_pressure().used_pct / 100.0) - sample_python_host_memory_mib()
                    peak_paused_gpu_mib = (sample_gpu_memory_pressure(int(args.gpu_device_index)).total_mib * sample_gpu_memory_pressure(int(args.gpu_device_index)).used_pct / 100.0) - sample_python_gpu_memory_mib(int(args.gpu_device_index))
                pressure_backoff_pending = True
                pressure_backoff_reason = pause_reason
                batch_backoff_state.update(
                    {
                        "batch_scale": batch_scale,
                        "pressure_backoff_pending": pressure_backoff_pending,
                        "last_backoff_reason": pressure_backoff_reason,
                    }
                )
                write_batch_backoff_state(run_root, batch_backoff_state)
                continue
            failure_counts[job.child_root] += 1
            if active_job.device_mode == "cuda" and child_log_indicates_cuda_oom(active_job.log_path):
                logger.log_console(
                    f"[GPU_OOM] retry_forever task={job.task_name} phase={job.phase_name} params={job.parameter_count} "
                    f"exit={code} retry_count={failure_counts[job.child_root]}"
                )
                mark_child_retrying(job.child_root, job, int(code or 0), active_job.cmd, failure_counts[job.child_root])
                gpu_peers = [
                    entry
                    for peer_proc, entry in active.items()
                    if peer_proc is not proc and entry.device_mode == "cuda" and not entry.pause_requested
                ]
                if gpu_peers:
                    largest_gpu = max(gpu_peers, key=lambda entry: (entry.job.parameter_count, entry.job.depth, entry.job.task_name))
                    largest_gpu.pause_requested = True
                    largest_gpu.pause_reason = "peer_cuda_oom"
                    mark_child_pause_requested(largest_gpu.job.child_root, largest_gpu.job, "peer_cuda_oom")
                    logger.log_console(
                        f"[GPU_OOM] request_pause_peer task={largest_gpu.job.task_name} phase={largest_gpu.job.phase_name} "
                        f"params={largest_gpu.job.parameter_count} because_failed_task={job.task_name} because_failed_phase={job.phase_name}"
                    )
                    terminate_child_process(next(peer_proc for peer_proc, entry in active.items() if entry is largest_gpu))
                pending.appendleft(job)
                if launches_enabled:
                    launches_enabled = False
                    peak_paused_host_mib = (sample_host_memory_pressure().total_mib * sample_host_memory_pressure().used_pct / 100.0) - sample_python_host_memory_mib()
                    peak_paused_gpu_mib = (sample_gpu_memory_pressure(int(args.gpu_device_index)).total_mib * sample_gpu_memory_pressure(int(args.gpu_device_index)).used_pct / 100.0) - sample_python_gpu_memory_mib(int(args.gpu_device_index))
                pressure_backoff_pending = True
                pressure_backoff_reason = "cuda_oom"
                batch_backoff_state.update(
                    {
                        "batch_scale": batch_scale,
                        "pressure_backoff_pending": pressure_backoff_pending,
                        "last_backoff_reason": pressure_backoff_reason,
                    }
                )
                write_batch_backoff_state(run_root, batch_backoff_state)
                continue
            logger.log_console(
                f"[TASK] retry_forever task={job.task_name} phase={job.phase_name} params={job.parameter_count} "
                f"exit={code} retry_count={failure_counts[job.child_root]}"
            )
            mark_child_retrying(job.child_root, job, int(code or 0), active_job.cmd, failure_counts[job.child_root])
            pending.appendleft(job)
            if launches_enabled:
                launches_enabled = False
                peak_paused_host_mib = (sample_host_memory_pressure().total_mib * sample_host_memory_pressure().used_pct / 100.0) - sample_python_host_memory_mib()
                peak_paused_gpu_mib = (sample_gpu_memory_pressure(int(args.gpu_device_index)).total_mib * sample_gpu_memory_pressure(int(args.gpu_device_index)).used_pct / 100.0) - sample_python_gpu_memory_mib(int(args.gpu_device_index))
            pressure_backoff_pending = True
            pressure_backoff_reason = "retry"
            batch_backoff_state.update(
                {
                    "batch_scale": batch_scale,
                    "pressure_backoff_pending": pressure_backoff_pending,
                    "last_backoff_reason": pressure_backoff_reason,
                }
            )
            write_batch_backoff_state(run_root, batch_backoff_state)
            continue

        for proc in finished:
            entry = active.pop(proc, None)
            if entry is not None:
                free_slots.add(entry.slot_index)
                if not launches_enabled:
                    if entry.pause_requested:
                        peak_paused_host_mib = (sample_host_memory_pressure().total_mib * sample_host_memory_pressure().used_pct / 100.0) - sample_python_host_memory_mib()
                        peak_paused_gpu_mib = (sample_gpu_memory_pressure(int(args.gpu_device_index)).total_mib * sample_gpu_memory_pressure(int(args.gpu_device_index)).used_pct / 100.0) - sample_python_gpu_memory_mib(int(args.gpu_device_index))
                    else:
                        logger.log_console(f"[STATE] admission gate reopened by genuine process completion")
                        launches_enabled = True
                        pressure_backoff_pending = False

        if not active and pending and pressure_backoff_pending:
            previous_scale = batch_scale
            batch_backoff_state = apply_batch_backoff(
                run_root,
                batch_backoff_state,
                float(getattr(args, "batch_backoff_factor", 0.5)),
                str(pressure_backoff_reason or "pressure_stall"),
            )
            batch_scale = float(batch_backoff_state.get("batch_scale", previous_scale))
            pressure_backoff_pending = False
            launches_enabled = True
            launch_sample_hold_until = time.time() + launch_sample_delay_sec
            logger.log_console(
                f"[STATE] batch_backoff reason={pressure_backoff_reason or 'pressure_stall'} "
                f"batch_scale={batch_scale:.6f} previous_scale={previous_scale:.6f}"
            )
            continue

        pressure = sample_host_memory_pressure()
        gpu_pressure = sample_gpu_memory_pressure(int(args.gpu_device_index))
        if active and gpu_available and gpu_pressure.total_mib > 0 and gpu_pressure.used_pct > float(args.gpu_memory_pressure_limit_pct):
            pausable_gpu = [entry for entry in active.values() if entry.device_mode == "cuda" and not entry.pause_requested]
            if pausable_gpu:
                largest_gpu = max(pausable_gpu, key=lambda entry: (entry.job.parameter_count, entry.job.depth, entry.job.task_name))
                largest_gpu.pause_requested = True
                largest_gpu.pause_reason = "gpu_memory_pressure"
                mark_child_pause_requested(largest_gpu.job.child_root, largest_gpu.job, "gpu_memory_pressure")
                logger.log_console(
                    f"[PRESSURE] request_pause_gpu task={largest_gpu.job.task_name} phase={largest_gpu.job.phase_name} "
                    f"params={largest_gpu.job.parameter_count} gpu_used_pct={gpu_pressure.used_pct:.2f} "
                    f"gpu_used_mib={gpu_pressure.used_mib}/{gpu_pressure.total_mib}"
                )
                terminate_child_process(next(proc for proc, entry in active.items() if entry is largest_gpu))
                continue

        current_host_mib = pressure.total_mib * pressure.used_pct / 100.0
        current_gpu_mib = gpu_pressure.total_mib * gpu_pressure.used_pct / 100.0

        if not launches_enabled:
            effective_host_mib = current_host_mib - sample_python_host_memory_mib()
            effective_gpu_mib = current_gpu_mib - sample_python_gpu_memory_mib(int(args.gpu_device_index))
            peak_paused_host_mib = max(peak_paused_host_mib, effective_host_mib)
            peak_paused_gpu_mib = max(peak_paused_gpu_mib, effective_gpu_mib)
            host_drop = peak_paused_host_mib - effective_host_mib
            gpu_drop = peak_paused_gpu_mib - effective_gpu_mib
            
            if host_drop >= 500.0 or gpu_drop >= 500.0 or (pressure.used_pct <= float(args.host_ram_resume_pct) and gpu_pressure.used_pct <= float(args.gpu_memory_resume_pct)):
                logger.log_console(
                    f"[STATE] admission gate reopened by >500MiB external drop or thresholds met "
                    f"host_drop={host_drop:.1f} gpu_drop={gpu_drop:.1f} host_used_pct={pressure.used_pct:.2f}"
                )
                launches_enabled = True
                pressure_backoff_pending = False

        if active and pressure.used_pct > float(args.host_ram_pressure_limit_pct):
            pausable = [entry for entry in active.values() if not entry.pause_requested]
            if pausable:
                largest = max(pausable, key=lambda entry: (entry.job.parameter_count, entry.job.depth, entry.job.task_name))
                largest.pause_requested = True
                largest.pause_reason = "host_ram_pressure"
                mark_child_pause_requested(largest.job.child_root, largest.job, "host_ram_pressure")
                logger.log_console(
                    f"[PRESSURE] request_pause task={largest.job.task_name} phase={largest.job.phase_name} "
                    f"params={largest.job.parameter_count} used_pct={pressure.used_pct:.2f} "
                    f"avail_mib={pressure.available_mib}/{pressure.total_mib}"
                )
                terminate_child_process(next(proc for proc, entry in active.items() if entry is largest))
                if launches_enabled:
                    launches_enabled = False
                    peak_paused_host_mib = (sample_host_memory_pressure().total_mib * sample_host_memory_pressure().used_pct / 100.0) - sample_python_host_memory_mib()
                    peak_paused_gpu_mib = (sample_gpu_memory_pressure(int(args.gpu_device_index)).total_mib * sample_gpu_memory_pressure(int(args.gpu_device_index)).used_pct / 100.0) - sample_python_gpu_memory_mib(int(args.gpu_device_index))
                continue

        if time.time() < launch_sample_hold_until:
            time.sleep(max(0.1, float(args.pressure_poll_interval_sec)))
            continue

        under_active_limit = len(active) < int(active_limit)
        active_gpu_jobs = sum(1 for entry in active.values() if entry.device_mode == "cuda")
        chosen_device_mode = choose_device_mode(pressure, gpu_pressure, active_gpu_jobs)
        can_launch_now = launches_enabled
        if pending and under_active_limit and free_slots and can_launch_now and chosen_device_mode is not None:
            job = pending.popleft()
            if child_completed(job.child_root, job.task_name):
                mark_child_completed(job.child_root, job)
                logger.log_console(
                    f"[TASK] already_complete task={job.task_name} phase={job.phase_name} params={job.parameter_count}"
                )
                continue
            device_mode = chosen_device_mode or "cpu"
            effective_batch_size = scale_batch_size(
                task_base_batch_sizes.get(job.task_name, int(args.batch_size) if int(args.batch_size) > 0 else 1),
                batch_scale,
            )
            cmd = build_worker_command(
                args=args,
                task_name=job.task_name,
                architecture=job.architecture,
                child_run_root=job.child_root,
                device_mode=device_mode,
                batch_size=effective_batch_size,
            )
            launches_total += 1
            mark_child_running(job.child_root, job, cmd, launches_total, effective_batch_size)
            log_path = child_process_log_path(job.child_root)
            slot_index = min(free_slots)
            free_slots.remove(slot_index)
            proc, log_handle = launch_child_process(
                cmd,
                env=launcher_child_env(
                    build_child_env(device_mode),
                    concurrency_hint=int(active_limit),
                    job_key=f"{job.task_name}:{job.phase_name}:{job.child_root}",
                    affinity_slot=slot_index,
                ),
                log_path=log_path,
            )
            active[proc] = ActiveChildJob(
                job=job,
                cmd=cmd,
                device_mode=device_mode,
                log_path=log_path,
                slot_index=slot_index,
                log_handle=log_handle,
                pause_requested=False,
                launch_count=launches_total,
            )
            logger.log_console(
                f"[TASK] launch task={job.task_name} phase={job.phase_name} params={job.parameter_count} "
                f"device={device_mode} batch_size={effective_batch_size} batch_scale={batch_scale:.6f} "
                f"host_used_pct={pressure.used_pct:.2f} avail_mib={pressure.available_mib}/{pressure.total_mib} "
                f"gpu_used_pct={gpu_pressure.used_pct:.2f} gpu_used_mib={gpu_pressure.used_mib}/{gpu_pressure.total_mib} "
                f"active={len(active)}/{active_limit}"
            )
            launch_sample_hold_until = time.time() + launch_sample_delay_sec
            continue

        if active or pending:
            time.sleep(max(0.1, float(args.pressure_poll_interval_sec)))

    task_reports: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    for task_name in tasks:
        report = aggregate_task(task_name, run_root / task_name, task_child_roots.get(task_name, []))
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
            "tasks": list(tasks),
            "scheduler": "pressure_aware",
            "param_band": list(stl.normalize_param_band(getattr(args, "param_band", None))) if getattr(args, "param_band", None) else None,
            "job_start_index": int(getattr(args, "job_start_index", 0) or 0),
            "job_limit": int(getattr(args, "job_limit", 0) or 0),
            "host_ram_pressure_limit_pct": float(args.host_ram_pressure_limit_pct),
            "host_ram_resume_pct": float(args.host_ram_resume_pct),
            "gpu_memory_pressure_limit_pct": float(args.gpu_memory_pressure_limit_pct),
            "gpu_memory_resume_pct": float(args.gpu_memory_resume_pct),
            "gpu_device_index": int(args.gpu_device_index),
            "max_active_jobs": int(active_limit),
            "repeat_count": int(args.repeat_count),
            "reports": task_reports,
        },
    )
    return task_reports


def run_gpu_first(args: argparse.Namespace, run_root: Path, tasks: Sequence[str], logger: ContinuousLogger) -> List[Dict[str, Any]]:
    """GPU-first dual-gate scheduler.

    Two independent admission gates:
      gpu_launches_enabled  — closes on GPU pressure/OOM/failure; reopens only when a GPU
                              child genuinely completes or GPU VRAM drops >=500 MiB.
      cpu_launches_enabled  — closes on host-RAM pressure/CPU failure; reopens on any genuine
                              completion, RAM drop >=500 MiB, or thresholds met.
    Each gate has its own post-launch hold timer.  The job queue, manifest, and batch scale
    are shared with pressure_aware — run roots are fully interchangeable.
    """
    jobs_by_task = build_task_jobs(args, tasks, run_root)
    task_base_batch_sizes = resolve_task_base_batch_sizes(args, tasks)
    sorted_jobs = sorted(
        (job for jobs in jobs_by_task.values() for job in jobs), key=pending_job_sort_key
    )
    pending_jobs = slice_pending_jobs(
        sorted_jobs,
        int(getattr(args, "job_start_index", 0) or 0),
        int(getattr(args, "job_limit", 0) or 0),
    )
    pending: Deque[ChildJob] = deque(pending_jobs)
    task_child_roots: Dict[str, List[Path]] = {
        tn: [j.child_root for j in jobs] for tn, jobs in jobs_by_task.items()
    }
    active: Dict[subprocess.Popen[Any], ActiveChildJob] = {}
    active_limit = active_job_limit(args, len(pending))
    slot_count = max(1, int(active_limit))
    free_slots = set(range(slot_count))
    failure_counts: Dict[Path, int] = defaultdict(int)
    launches_total = 0
    gpu_available = bool(torch.cuda.is_available())
    launch_sample_delay_sec = max(0.0, float(getattr(args, "post_launch_sample_delay_sec", 30.0)))

    # Dual gates
    gpu_launches_enabled = True
    cpu_launches_enabled = True
    # Per-resource post-launch hold timers
    gpu_hold_until = 0.0
    cpu_hold_until = 0.0
    # Peak memory for 500 MiB drop heuristic (non-Python process memory)
    gpu_peak_vram_mib = 0.0
    cpu_peak_host_mib = 0.0

    batch_backoff_state = load_batch_backoff_state(run_root)
    batch_scale = float(batch_backoff_state.get("batch_scale", 1.0))
    pressure_backoff_pending = bool(batch_backoff_state.get("pressure_backoff_pending", False))
    pressure_backoff_reason: Optional[str] = batch_backoff_state.get("last_backoff_reason")

    def _build_env(device_mode: str) -> Dict[str, str]:
        env = os.environ.copy()
        if device_mode == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif device_mode == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_device_index))
        return env

    def _snap_gpu() -> float:
        gp = sample_gpu_memory_pressure(int(args.gpu_device_index))
        return float(gp.total_mib * gp.used_pct / 100.0)

    def _snap_host_net() -> float:
        h = sample_host_memory_pressure()
        return float(h.total_mib * h.used_pct / 100.0) - sample_python_host_memory_mib()

    def _save_backoff() -> None:
        batch_backoff_state.update({
            "batch_scale": batch_scale,
            "pressure_backoff_pending": pressure_backoff_pending,
            "last_backoff_reason": pressure_backoff_reason,
        })
        write_batch_backoff_state(run_root, batch_backoff_state)

    while pending or active:
        # ── 1. Poll all active children ──────────────────────────────────────
        finished: List[subprocess.Popen[Any]] = []
        for proc, active_job in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            finished.append(proc)
            job = active_job.job
            close_child_log(active_job.log_handle)

            if child_completed(job.child_root, job.task_name):
                mark_child_completed(job.child_root, job)
                logger.log_console(
                    f"[TASK] completed task={job.task_name} phase={job.phase_name} "
                    f"params={job.parameter_count} device={active_job.device_mode}"
                )
                # GPU completion → reopen GPU gate
                if active_job.device_mode == "cuda" and not gpu_launches_enabled:
                    gpu_launches_enabled = True
                    logger.log_console(f"[GPU_GATE] reopened by GPU job completion task={job.task_name}")
                # Any genuine completion → reopen CPU gate
                if not cpu_launches_enabled:
                    cpu_launches_enabled = True
                    logger.log_console(f"[CPU_GATE] reopened by job completion task={job.task_name}")
                pressure_backoff_pending = False
                pressure_backoff_reason = None
                _save_backoff()
                continue

            if active_job.pause_requested:
                pause_reason = str(active_job.pause_reason or "pressure")
                mark_child_paused(job.child_root, job, int(code or 0), pause_reason)
                pending.appendleft(job)
                logger.log_console(
                    f"[PRESSURE] paused task={job.task_name} phase={job.phase_name} "
                    f"device={active_job.device_mode} reason={pause_reason} exit={code}"
                )
                if active_job.device_mode == "cuda":
                    if gpu_launches_enabled:
                        gpu_launches_enabled = False
                        gpu_peak_vram_mib = _snap_gpu()
                        logger.log_console(
                            f"[GPU_GATE] closed reason={pause_reason} peak_vram={gpu_peak_vram_mib:.0f} MiB"
                        )
                else:
                    if cpu_launches_enabled:
                        cpu_launches_enabled = False
                        cpu_peak_host_mib = _snap_host_net()
                        logger.log_console(
                            f"[CPU_GATE] closed reason={pause_reason} peak_host_net={cpu_peak_host_mib:.0f} MiB"
                        )
                pressure_backoff_pending = True
                pressure_backoff_reason = pause_reason
                _save_backoff()
                continue

            # CUDA OOM
            failure_counts[job.child_root] += 1
            if active_job.device_mode == "cuda" and child_log_indicates_cuda_oom(active_job.log_path):
                logger.log_console(
                    f"[GPU_OOM] task={job.task_name} phase={job.phase_name} "
                    f"exit={code} retries={failure_counts[job.child_root]}"
                )
                mark_child_retrying(job.child_root, job, int(code or 0), active_job.cmd, failure_counts[job.child_root])
                if gpu_launches_enabled:
                    gpu_launches_enabled = False
                    gpu_peak_vram_mib = _snap_gpu()
                    logger.log_console(f"[GPU_GATE] closed by CUDA OOM peak_vram={gpu_peak_vram_mib:.0f} MiB")
                gpu_peers = [
                    entry for peer, entry in active.items()
                    if peer is not proc and entry.device_mode == "cuda" and not entry.pause_requested
                ]
                if gpu_peers:
                    largest = max(gpu_peers, key=lambda e: (e.job.parameter_count, e.job.depth, e.job.task_name))
                    largest.pause_requested = True
                    largest.pause_reason = "peer_cuda_oom"
                    mark_child_pause_requested(largest.job.child_root, largest.job, "peer_cuda_oom")
                    terminate_child_process(next(p for p, e in active.items() if e is largest))
                    logger.log_console(
                        f"[GPU_OOM] paused_peer task={largest.job.task_name} phase={largest.job.phase_name}"
                    )
                pending.appendleft(job)
                pressure_backoff_pending = True
                pressure_backoff_reason = "cuda_oom"
                _save_backoff()
                continue

            # Generic failure (CPU or GPU)
            mark_child_retrying(job.child_root, job, int(code or 0), active_job.cmd, failure_counts[job.child_root])
            logger.log_console(
                f"[TASK] retry_forever task={job.task_name} phase={job.phase_name} "
                f"device={active_job.device_mode} exit={code} retries={failure_counts[job.child_root]}"
            )
            pending.appendleft(job)
            if active_job.device_mode == "cuda":
                if gpu_launches_enabled:
                    gpu_launches_enabled = False
                    gpu_peak_vram_mib = _snap_gpu()
                    logger.log_console(f"[GPU_GATE] closed by GPU failure task={job.task_name}")
            else:
                if cpu_launches_enabled:
                    cpu_launches_enabled = False
                    cpu_peak_host_mib = _snap_host_net()
                    logger.log_console(f"[CPU_GATE] closed by CPU failure task={job.task_name}")
            pressure_backoff_pending = True
            pressure_backoff_reason = "retry"
            _save_backoff()

        # ── 2. Free slots; reopen gates on genuine non-pause process exit ────
        for proc in finished:
            entry = active.pop(proc, None)
            if entry is None:
                continue
            free_slots.add(entry.slot_index)
            if entry.pause_requested:
                # Refresh peak after the evicted child's memory returns
                if entry.device_mode == "cuda":
                    gpu_peak_vram_mib = _snap_gpu()
                else:
                    cpu_peak_host_mib = _snap_host_net()
            else:
                # Genuine unexpected exit (not completed, not paused)
                if not cpu_launches_enabled:
                    cpu_launches_enabled = True
                    pressure_backoff_pending = False
                    logger.log_console("[CPU_GATE] reopened by genuine process exit")
                if entry.device_mode == "cuda" and not gpu_launches_enabled:
                    gpu_launches_enabled = True
                    logger.log_console("[GPU_GATE] reopened by genuine GPU process exit")

        # ── 3. Batch backoff when queue fully drained under pressure ─────────
        if not active and pending and pressure_backoff_pending:
            previous_scale = batch_scale
            batch_backoff_state = apply_batch_backoff(
                run_root, batch_backoff_state,
                float(getattr(args, "batch_backoff_factor", 0.5)),
                str(pressure_backoff_reason or "pressure_stall"),
            )
            batch_scale = float(batch_backoff_state.get("batch_scale", previous_scale))
            pressure_backoff_pending = False
            gpu_launches_enabled = True
            cpu_launches_enabled = True
            gpu_hold_until = time.time() + launch_sample_delay_sec
            cpu_hold_until = time.time() + launch_sample_delay_sec
            logger.log_console(
                f"[STATE] batch_backoff reason={pressure_backoff_reason or 'pressure_stall'} "
                f"batch_scale={batch_scale:.6f} previous_scale={previous_scale:.6f}"
            )
            continue

        # ── 4. Sample pressure ───────────────────────────────────────────────
        pressure = sample_host_memory_pressure()
        gpu_pressure = sample_gpu_memory_pressure(int(args.gpu_device_index))
        current_gpu_mib = float(gpu_pressure.total_mib * gpu_pressure.used_pct / 100.0)
        current_host_net = float(pressure.total_mib * pressure.used_pct / 100.0) - sample_python_host_memory_mib()

        # ── 5. GPU VRAM over limit → evict largest GPU child, close GPU gate ─
        if (active and gpu_available and gpu_pressure.total_mib > 0
                and gpu_pressure.used_pct > float(args.gpu_memory_pressure_limit_pct)):
            gpu_pausable = [e for e in active.values() if e.device_mode == "cuda" and not e.pause_requested]
            if gpu_pausable:
                victim = max(gpu_pausable, key=lambda e: (e.job.parameter_count, e.job.depth, e.job.task_name))
                victim.pause_requested = True
                victim.pause_reason = "gpu_memory_pressure"
                mark_child_pause_requested(victim.job.child_root, victim.job, "gpu_memory_pressure")
                if gpu_launches_enabled:
                    gpu_launches_enabled = False
                    gpu_peak_vram_mib = current_gpu_mib
                    logger.log_console(
                        f"[GPU_GATE] closed by VRAM pressure gpu_used_pct={gpu_pressure.used_pct:.2f} "
                        f"gpu_used_mib={gpu_pressure.used_mib}/{gpu_pressure.total_mib}"
                    )
                logger.log_console(
                    f"[GPU_PRESSURE] evict task={victim.job.task_name} phase={victim.job.phase_name} "
                    f"params={victim.job.parameter_count}"
                )
                terminate_child_process(next(p for p, e in active.items() if e is victim))
                continue

        # ── 6. Host RAM over limit → evict largest ANY child, close CPU gate ─
        if active and pressure.used_pct > float(args.host_ram_pressure_limit_pct):
            pausable = [e for e in active.values() if not e.pause_requested]
            if pausable:
                victim = max(pausable, key=lambda e: (e.job.parameter_count, e.job.depth, e.job.task_name))
                victim.pause_requested = True
                victim.pause_reason = "host_ram_pressure"
                mark_child_pause_requested(victim.job.child_root, victim.job, "host_ram_pressure")
                if cpu_launches_enabled:
                    cpu_launches_enabled = False
                    cpu_peak_host_mib = current_host_net
                    logger.log_console(
                        f"[CPU_GATE] closed by RAM pressure ram_used_pct={pressure.used_pct:.2f} "
                        f"avail_mib={pressure.available_mib}/{pressure.total_mib}"
                    )
                logger.log_console(
                    f"[RAM_PRESSURE] evict task={victim.job.task_name} phase={victim.job.phase_name} "
                    f"params={victim.job.parameter_count} device={victim.device_mode}"
                )
                terminate_child_process(next(p for p, e in active.items() if e is victim))
                continue

        # ── 7. Gate reopen: GPU — 500 MiB drop or below resume threshold ────
        if not gpu_launches_enabled:
            gpu_drop = gpu_peak_vram_mib - current_gpu_mib
            if gpu_drop >= 500.0 or gpu_pressure.used_pct <= float(args.gpu_memory_resume_pct):
                gpu_launches_enabled = True
                logger.log_console(
                    f"[GPU_GATE] reopened gpu_drop={gpu_drop:.1f} MiB "
                    f"gpu_used_pct={gpu_pressure.used_pct:.2f}"
                )

        # ── 8. Gate reopen: CPU — 500 MiB drop or below resume thresholds ───
        if not cpu_launches_enabled:
            host_drop = cpu_peak_host_mib - current_host_net
            if host_drop >= 500.0 or (
                pressure.used_pct <= float(args.host_ram_resume_pct)
                and gpu_pressure.used_pct <= float(args.gpu_memory_resume_pct)
            ):
                cpu_launches_enabled = True
                logger.log_console(
                    f"[CPU_GATE] reopened host_drop={host_drop:.1f} MiB "
                    f"ram_used_pct={pressure.used_pct:.2f}"
                )

        # ── 9. Launch decision: GPU first, CPU fallback ──────────────────────
        now = time.time()
        active_gpu_jobs = sum(1 for e in active.values() if e.device_mode == "cuda")
        gpu_limit = active_gpu_job_limit(args)
        gpu_count_ok = gpu_limit <= 0 or active_gpu_jobs < gpu_limit
        under_limit = len(active) < int(active_limit)

        can_gpu = (
            gpu_launches_enabled
            and now >= gpu_hold_until
            and gpu_available
            and gpu_pressure.total_mib > 0
            and pressure.used_pct <= float(args.host_ram_resume_pct)
            and gpu_pressure.used_pct <= float(args.gpu_memory_resume_pct)
            and gpu_count_ok
        )
        can_cpu = (
            cpu_launches_enabled
            and now >= cpu_hold_until
            and pressure.used_pct <= float(args.host_ram_resume_pct)
        )

        if pending and under_limit and free_slots:
            if can_gpu:
                device_mode = "cuda"
            elif can_cpu:
                device_mode = "cpu"
            else:
                time.sleep(max(0.1, float(args.pressure_poll_interval_sec)))
                continue

            job = pending.popleft()
            if child_completed(job.child_root, job.task_name):
                mark_child_completed(job.child_root, job)
                logger.log_console(
                    f"[TASK] already_complete task={job.task_name} phase={job.phase_name}"
                )
                continue

            effective_batch_size = scale_batch_size(
                task_base_batch_sizes.get(job.task_name, int(args.batch_size) if int(args.batch_size) > 0 else 1),
                batch_scale,
            )
            cmd = build_worker_command(
                args=args, task_name=job.task_name, architecture=job.architecture,
                child_run_root=job.child_root, device_mode=device_mode, batch_size=effective_batch_size,
            )
            launches_total += 1
            mark_child_running(job.child_root, job, cmd, launches_total, effective_batch_size)
            log_path = child_process_log_path(job.child_root)
            slot_index = min(free_slots)
            free_slots.remove(slot_index)
            proc, log_handle = launch_child_process(
                cmd,
                env=launcher_child_env(
                    _build_env(device_mode),
                    concurrency_hint=int(active_limit),
                    job_key=f"{job.task_name}:{job.phase_name}:{job.child_root}",
                    affinity_slot=slot_index,
                ),
                log_path=log_path,
            )
            active[proc] = ActiveChildJob(
                job=job, cmd=cmd, device_mode=device_mode, log_path=log_path,
                slot_index=slot_index, log_handle=log_handle,
                pause_requested=False, launch_count=launches_total,
            )
            logger.log_console(
                f"[TASK] launch task={job.task_name} phase={job.phase_name} "
                f"params={job.parameter_count} device={device_mode} "
                f"batch_size={effective_batch_size} batch_scale={batch_scale:.6f} "
                f"ram_used_pct={pressure.used_pct:.2f} avail_mib={pressure.available_mib}/{pressure.total_mib} "
                f"gpu_used_pct={gpu_pressure.used_pct:.2f} gpu_mib={gpu_pressure.used_mib}/{gpu_pressure.total_mib} "
                f"active={len(active)}/{active_limit} gpu_gate={gpu_launches_enabled} cpu_gate={cpu_launches_enabled}"
            )
            if device_mode == "cuda":
                gpu_hold_until = time.time() + launch_sample_delay_sec
            else:
                cpu_hold_until = time.time() + launch_sample_delay_sec
            continue

        if active or pending:
            time.sleep(max(0.1, float(args.pressure_poll_interval_sec)))

    # Aggregate results (same as pressure_aware)
    task_reports: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    for task_name in tasks:
        report = aggregate_task(task_name, run_root / task_name, task_child_roots.get(task_name, []))
        task_reports.append(report)
        comparison_rows.extend(report.get("comparisons", []))
    if comparison_rows:
        rg.write_csv(
            run_root / "comparison_summary.csv",
            comparison_rows,
            fieldnames=[
                "task", "repeat", "ablation_phase", "ablation_architecture",
                "ablation_parameter_count", "ablation_best_val", "reference_kind",
                "reference_phase", "reference_architecture", "reference_best_val",
                "winner", "winner_value",
            ],
        )
    rg.write_json(
        run_root / "comparison_summary.json",
        {
            "tasks": list(tasks),
            "scheduler": "gpu_first",
            "param_band": list(stl.normalize_param_band(getattr(args, "param_band", None))) if getattr(args, "param_band", None) else None,
            "job_start_index": int(getattr(args, "job_start_index", 0) or 0),
            "job_limit": int(getattr(args, "job_limit", 0) or 0),
            "host_ram_pressure_limit_pct": float(args.host_ram_pressure_limit_pct),
            "host_ram_resume_pct": float(args.host_ram_resume_pct),
            "gpu_memory_pressure_limit_pct": float(args.gpu_memory_pressure_limit_pct),
            "gpu_memory_resume_pct": float(args.gpu_memory_resume_pct),
            "gpu_device_index": int(args.gpu_device_index),
            "max_active_jobs": int(active_limit),
            "repeat_count": int(args.repeat_count),
            "reports": task_reports,
        },
    )
    return task_reports


def main() -> None:
    args = parse_args()
    args.concurrency = resolve_concurrency(args)
    os.environ["TABULAR_CPU_JOB_CONCURRENCY"] = str(int(args.concurrency))
    bootstrap_runtime("run_stl_ablation_parallel")
    if float(args.host_ram_resume_pct) > float(args.host_ram_pressure_limit_pct):
        raise SystemExit("--host-ram-resume-pct must be <= --host-ram-pressure-limit-pct")
    if float(args.gpu_memory_resume_pct) > float(args.gpu_memory_pressure_limit_pct):
        raise SystemExit("--gpu-memory-resume-pct must be <= --gpu-memory-pressure-limit-pct")
    if not (0.0 < float(getattr(args, "batch_backoff_factor", 0.5)) < 1.0):
        raise SystemExit("--batch-backoff-factor must be between 0 and 1")
    tasks = [str(t).lower() for t in args.tasks]
    architectures = stl.build_architectures(args)
    if not architectures:
        raise SystemExit("No architectures requested.")
    param_band = stl.normalize_param_band(getattr(args, "param_band", None))

    if args.run_root:
        run_root = Path(args.run_root)
    else:
        band_label = stl.param_band_label(param_band)
        suffix = f"_{band_label}" if band_label else ""
        run_root = Path(args.results_dir) / f"stl_ablation_parallel{suffix}_{rg.now_stamp()}"
    run_root.mkdir(parents=True, exist_ok=True)
    logger = ContinuousLogger(run_root, "stl_ablation_parallel", "stl_ablation_parallel")
    logger.log_console(f"Run root: {run_root}")
    logger.log_console(f"Tasks: {tasks}")
    logger.log_console(f"Architectures: {[rg.format_architecture_for_report(a) for a in architectures]}")
    if param_band is not None:
        logger.log_console(f"Parameter decade band: {list(param_band)}")
    logger.log_console(f"Repeat count: {int(args.repeat_count)}")
    logger.log_console(f"Scheduler: {args.scheduler}")
    logger.log_console(f"Concurrency: {int(args.concurrency)}")
    logger.log_console(f"Job start index: {int(getattr(args, 'job_start_index', 0) or 0)}")
    logger.log_console(f"Job limit: {int(getattr(args, 'job_limit', 0) or 0)}")
    if getattr(args, "concurrency_file", None):
        logger.log_console(f"Concurrency file: {args.concurrency_file}")
    if args.scheduler == "pressure_aware":
        logger.log_console(f"Host RAM pressure limit pct: {float(args.host_ram_pressure_limit_pct):.2f}")
        logger.log_console(f"Host RAM resume pct: {float(args.host_ram_resume_pct):.2f}")
        logger.log_console(f"GPU memory pressure limit pct: {float(args.gpu_memory_pressure_limit_pct):.2f}")
        logger.log_console(f"GPU memory resume pct: {float(args.gpu_memory_resume_pct):.2f}")
        logger.log_console(f"GPU device index: {int(args.gpu_device_index)}")
        logger.log_console(f"Max active jobs: {int(args.max_active_jobs)}")
        logger.log_console(f"Max retries per job: {int(args.max_retries_per_job)}")
        logger.log_console(f"Batch backoff factor: {float(args.batch_backoff_factor):.3f}")
        logger.log_console(f"Batch backoff scale: {float(load_batch_backoff_state(run_root).get('batch_scale', 1.0)):.6f}")
    logger.log_console(f"Source run root: {args.source_run_root}")
    logger.log_console(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    if args.scheduler in ("pressure_aware", "gpu_first"):
        if args.scheduler == "gpu_first":
            logger.log_console("Scheduler: gpu_first (dual-gate: GPU-first, CPU fallback)")
            logger.log_console(f"GPU gate: reopen on GPU job completion or >500 MiB VRAM drop")
            logger.log_console(f"CPU gate: reopen on any completion, >500 MiB RAM drop, or thresholds met")
            run_gpu_first(args, run_root, tasks, logger)
        else:
            run_pressure_aware(args, run_root, tasks, logger)
        logger.close()
        return

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
            "param_band": list(param_band) if param_band is not None else None,
            "scheduler": "fixed",
            "concurrency": int(args.concurrency),
            "concurrency_file": str(args.concurrency_file) if getattr(args, "concurrency_file", None) else None,
            "source_run_root": str(args.source_run_root),
            "repeat_count": int(args.repeat_count),
            "reports": task_reports,
        },
    )
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

    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Caught KeyboardInterrupt. Initiating global python purge (pkill -9 -f python)...")
        import subprocess, sys
        try:
            subprocess.run(["pkill", "-9", "-f", "python"])
        except Exception:
            pass
        sys.exit(130)
