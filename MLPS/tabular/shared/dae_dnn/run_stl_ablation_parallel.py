from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import torch

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
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=False)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
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
        "--concurrency-file",
        default=None,
        help="Optional text file containing the concurrency value to use instead of --concurrency.",
    )
    p.add_argument(
        "--scheduler",
        choices=["pressure_aware", "fixed"],
        default="pressure_aware",
        help="Use the pressure-aware global scheduler or the legacy fixed-slot task scheduler.",
    )
    p.add_argument(
        "--max-active-jobs",
        type=int,
        default=0,
        help="Hard cap for pressure-aware child jobs. 0 means use all visible logical CPUs as job lanes.",
    )
    p.add_argument(
        "--host-ram-pressure-limit-pct",
        type=float,
        default=90.0,
        help="Pause the largest active child when used host RAM exceeds this percentage.",
    )
    p.add_argument(
        "--host-ram-resume-pct",
        type=float,
        default=85.0,
        help="Only launch or relaunch a child when used host RAM is at or below this percentage.",
    )
    p.add_argument(
        "--pressure-poll-interval-sec",
        type=float,
        default=0.5,
        help="Polling interval for child completion and host RAM pressure checks.",
    )
    p.add_argument(
        "--pressure-settle-sec",
        type=float,
        default=1.0,
        help="Wait this long after each launch or pause so RAM pressure can settle before the next decision.",
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
        str(int(args.batch_size)),
        "--pin-memory" if bool(args.pin_memory) else "--no-pin-memory",
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


def mark_child_running(child_root: Path, job: ChildJob, cmd: Sequence[str], launch_count: int) -> None:
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


def build_task_jobs(args: argparse.Namespace, tasks: Sequence[str], run_root: Path) -> Dict[str, List[ChildJob]]:
    cfg = stl.make_cfg(args, list(tasks), run_root)
    base_architectures = stl.build_architectures(args)
    jobs_by_task: Dict[str, List[ChildJob]] = {}
    for task_name in tasks:
        task_root = run_root / task_name
        child_base = task_root / "_children"
        child_base.mkdir(parents=True, exist_ok=True)
        task = build_task(task_name, cfg.data_dir, 1, cfg.num_workers, cfg.seed, pin_memory=bool(args.pin_memory))
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
    return jobs_by_task


def active_job_limit(args: argparse.Namespace, job_count: int) -> int:
    limit = int(getattr(args, "max_active_jobs", 0) or 0)
    if limit <= 0:
        return max(1, min(int(job_count), int(detect_cpu_cores())))
    return max(1, int(limit))


def active_gpu_job_limit(args: argparse.Namespace) -> int:
    return max(0, int(getattr(args, "max_active_gpu_jobs", 0) or 0))


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
    pending: Deque[ChildJob] = deque(sorted((job for jobs in jobs_by_task.values() for job in jobs), key=pending_job_sort_key))
    task_child_roots: Dict[str, List[Path]] = {task_name: [job.child_root for job in jobs] for task_name, jobs in jobs_by_task.items()}
    active: Dict[subprocess.Popen[Any], ActiveChildJob] = {}
    slot_count = max(1, min(int(active_job_limit(args, len(pending))), int(detect_cpu_cores())))
    free_slots = set(range(slot_count))
    failure_counts: Dict[Path, int] = defaultdict(int)
    launches_total = 0
    active_limit = active_job_limit(args, len(pending))
    gpu_available = bool(torch.cuda.is_available())
    launches_enabled = True

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
                launches_enabled = True
                continue
            if active_job.pause_requested:
                logger.log_console(
                    f"[PRESSURE] paused task={job.task_name} phase={job.phase_name} params={job.parameter_count} exit={code}"
                )
                mark_child_paused(job.child_root, job, int(code or 0), "host_ram_pressure")
                pending.appendleft(job)
                launches_enabled = False
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
                    mark_child_pause_requested(largest_gpu.job.child_root, largest_gpu.job, "peer_cuda_oom")
                    logger.log_console(
                        f"[GPU_OOM] request_pause_peer task={largest_gpu.job.task_name} phase={largest_gpu.job.phase_name} "
                        f"params={largest_gpu.job.parameter_count} because_failed_task={job.task_name} because_failed_phase={job.phase_name}"
                    )
                    terminate_child_process(next(peer_proc for peer_proc, entry in active.items() if entry is largest_gpu))
                pending.appendleft(job)
                launches_enabled = False
                continue
            logger.log_console(
                f"[TASK] retry_forever task={job.task_name} phase={job.phase_name} params={job.parameter_count} "
                f"exit={code} retry_count={failure_counts[job.child_root]}"
            )
            mark_child_retrying(job.child_root, job, int(code or 0), active_job.cmd, failure_counts[job.child_root])
            pending.appendleft(job)
            launches_enabled = False
            continue

        for proc in finished:
            entry = active.pop(proc, None)
            if entry is not None:
                free_slots.add(entry.slot_index)

        pressure = sample_host_memory_pressure()
        gpu_pressure = sample_gpu_memory_pressure(int(args.gpu_device_index))
        if active and gpu_available and gpu_pressure.total_mib > 0 and gpu_pressure.used_pct > float(args.gpu_memory_pressure_limit_pct):
            pausable_gpu = [entry for entry in active.values() if entry.device_mode == "cuda" and not entry.pause_requested]
            if pausable_gpu:
                largest_gpu = max(pausable_gpu, key=lambda entry: (entry.job.parameter_count, entry.job.depth, entry.job.task_name))
                largest_gpu.pause_requested = True
                mark_child_pause_requested(largest_gpu.job.child_root, largest_gpu.job, "gpu_memory_pressure")
                logger.log_console(
                    f"[PRESSURE] request_pause_gpu task={largest_gpu.job.task_name} phase={largest_gpu.job.phase_name} "
                    f"params={largest_gpu.job.parameter_count} gpu_used_pct={gpu_pressure.used_pct:.2f} "
                    f"gpu_used_mib={gpu_pressure.used_mib}/{gpu_pressure.total_mib}"
                )
                terminate_child_process(next(proc for proc, entry in active.items() if entry is largest_gpu))
                time.sleep(max(0.0, float(args.pressure_settle_sec)))
                continue

        if active and pressure.used_pct > float(args.host_ram_pressure_limit_pct):
            pausable = [entry for entry in active.values() if not entry.pause_requested]
            if len(active) > 1 and pausable:
                largest = max(pausable, key=lambda entry: (entry.job.parameter_count, entry.job.depth, entry.job.task_name))
                largest.pause_requested = True
                mark_child_pause_requested(largest.job.child_root, largest.job, "host_ram_pressure")
                logger.log_console(
                    f"[PRESSURE] request_pause task={largest.job.task_name} phase={largest.job.phase_name} "
                    f"params={largest.job.parameter_count} used_pct={pressure.used_pct:.2f} "
                    f"avail_mib={pressure.available_mib}/{pressure.total_mib}"
                )
                terminate_child_process(next(proc for proc, entry in active.items() if entry is largest))
                time.sleep(max(0.0, float(args.pressure_settle_sec)))
                continue

        under_active_limit = len(active) < int(active_limit)
        active_gpu_jobs = sum(1 for entry in active.values() if entry.device_mode == "cuda")
        chosen_device_mode = choose_device_mode(pressure, gpu_pressure, active_gpu_jobs)
        can_launch_now = launches_enabled
        if pending and under_active_limit and free_slots and can_launch_now and (chosen_device_mode is not None or not active):
            job = pending.popleft()
            if child_completed(job.child_root, job.task_name):
                mark_child_completed(job.child_root, job)
                logger.log_console(
                    f"[TASK] already_complete task={job.task_name} phase={job.phase_name} params={job.parameter_count}"
                )
                continue
            device_mode = chosen_device_mode or "cpu"
            cmd = build_worker_command(
                args=args,
                task_name=job.task_name,
                architecture=job.architecture,
                child_run_root=job.child_root,
                device_mode=device_mode,
            )
            launches_total += 1
            mark_child_running(job.child_root, job, cmd, launches_total)
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
                f"device={device_mode} host_used_pct={pressure.used_pct:.2f} avail_mib={pressure.available_mib}/{pressure.total_mib} "
                f"gpu_used_pct={gpu_pressure.used_pct:.2f} gpu_used_mib={gpu_pressure.used_mib}/{gpu_pressure.total_mib} "
                f"active={len(active)}/{active_limit}"
            )
            time.sleep(max(0.0, float(args.pressure_settle_sec)))
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
    logger.log_console(f"Source run root: {args.source_run_root}")
    logger.log_console(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    if args.scheduler == "pressure_aware":
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
    main()
