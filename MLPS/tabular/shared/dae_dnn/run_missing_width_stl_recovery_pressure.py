from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch

from MLPS.tabular.shared.dae_dnn.tasks import build_task
from MLPS.tabular.shared.dae_dnn.runtime_tuning import bootstrap_runtime, detect_cpu_cores, launcher_child_env
from utils.adp_logging import ContinuousLogger

try:  # pragma: no cover - direct script execution
    import run_goliath as rg
    import run_stl_ablation as stl
    import run_stl_ablation_parallel as pressure
except ModuleNotFoundError:  # pragma: no cover - package import
    from MLPS.tabular.shared.dae_dnn import run_goliath as rg
    from MLPS.tabular.shared.dae_dnn import run_stl_ablation as stl
    from MLPS.tabular.shared.dae_dnn import run_stl_ablation_parallel as pressure


WIDTH_ONLY_PRESENT_TASKS = [
    "classification",
    "autoencoding",
    "generation",
    "denoising",
    "anomaly",
    "simulation",
]
WIDTH_ONLY_ALL_TASKS = [*WIDTH_ONLY_PRESENT_TASKS, "prediction"]
MISSING_ANOMALY_STL_LEAVES = [
    (6, 256),
    (8, 64),
    (8, 96),
    (8, 128),
    (8, 160),
    (8, 192),
    (8, 224),
    (8, 256),
    (10, 64),
    (10, 96),
    (10, 128),
    (10, 160),
    (10, 192),
    (10, 224),
    (10, 256),
]


@dataclass(frozen=True)
class RecoveryJob:
    name: str
    kind: str
    task: str
    root: Path
    command: Tuple[str, ...]
    parameter_count: int
    depth: int
    device_capable: bool = True


@dataclass
class ActiveRecoveryJob:
    job: RecoveryJob
    device_mode: str
    log_path: Path
    log_handle: Optional[Any]
    slot_index: int
    pause_requested: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pressure-aware recovery runner for missing width-only and small STL-grid results.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default="MLPS/tabular/shared/dae_dnn/results/recovery/missing_width_stl_v1")
    p.add_argument("--source-run-root", default="MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current")
    p.add_argument("--batch-size", type=int, default=186240)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--width-stage-margin-patience", type=int, default=10)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--repeat-count", type=int, default=5)
    p.add_argument("--width-depths", default="1,2,3,4,5,6")
    p.add_argument("--missing-present-task-repeats", default="2,3,4,5")
    p.add_argument("--prediction-repeats", default="1,2,3,4,5")
    p.add_argument("--host-ram-pressure-limit-pct", type=float, default=90.0)
    p.add_argument("--host-ram-resume-pct", type=float, default=85.0)
    p.add_argument("--gpu-memory-pressure-limit-pct", type=float, default=90.0)
    p.add_argument("--gpu-memory-resume-pct", type=float, default=85.0)
    p.add_argument("--gpu-device-index", type=int, default=0)
    p.add_argument("--pressure-poll-interval-sec", type=float, default=0.5)
    p.add_argument("--pressure-settle-sec", type=float, default=1.0)
    p.add_argument("--max-active-jobs", type=int, default=0, help="0 means no slot cap beyond pressure gates.")
    p.add_argument("--include-width-only", action="store_true", default=True)
    p.add_argument("--no-include-width-only", dest="include_width_only", action="store_false")
    p.add_argument("--include-small-stl", action="store_true", default=True)
    p.add_argument("--no-include-small-stl", dest="include_small_stl", action="store_false")
    p.add_argument("--dry-run", action="store_true", help="Write and print the sorted plan without launching children.")
    return p.parse_args()


def parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def repo_script(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def job_state_path(root: Path) -> Path:
    return root / "recovery_child_state.json"


def job_log_path(root: Path) -> Path:
    return root / "_recovery_child_process.log"


def load_json(path: Path) -> Dict[str, Any]:
    data = rg.load_json_if_exists(path)
    return data if isinstance(data, dict) else {}


def write_job_state(root: Path, payload: Dict[str, Any]) -> None:
    rg.write_json(job_state_path(root), payload)


def update_job_state(root: Path, updates: Dict[str, Any]) -> None:
    state = load_json(job_state_path(root))
    state.update(updates)
    rg.write_json(job_state_path(root), state)


def width_job_completed(job: RecoveryJob) -> bool:
    state = load_json(job.root / job.task / "task_state.json")
    if bool(state.get("completed", False)) and not bool(state.get("failed", False)):
        return True
    final_report = load_json(job.root / "final_report.json")
    return int((final_report.get("summary") or {}).get("num_tasks_completed", 0) or 0) >= 1


def stl_job_completed(job: RecoveryJob) -> bool:
    return pressure.child_completed(job.root, job.task)


def job_completed(job: RecoveryJob) -> bool:
    if job.kind == "width_only":
        return width_job_completed(job)
    if job.kind == "small_stl":
        return stl_job_completed(job)
    return False


def task_dims(task_name: str, args: argparse.Namespace, cache: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    if task_name not in cache:
        task = build_task(task_name, args.data_dir, 1, int(args.num_workers), int(args.seed), pin_memory=False)
        cache[task_name] = (int(task.in_dim), int(task.out_dim))
    return cache[task_name]


def estimated_width_params(task_name: str, depth: int, args: argparse.Namespace, dims_cache: Dict[str, Tuple[int, int]]) -> int:
    in_dim, out_dim = task_dims(task_name, args, dims_cache)
    width = min(int(args.max_width), int(stl.task_depth_max_width(task_name, int(depth))))
    return int(stl.parameter_count_for_architecture(in_dim, out_dim, [width] * int(depth), bool(args.use_bn)))


def stl_params(task_name: str, architecture: Sequence[int], args: argparse.Namespace, dims_cache: Dict[str, Tuple[int, int]]) -> int:
    in_dim, out_dim = task_dims(task_name, args, dims_cache)
    return int(stl.parameter_count_for_architecture(in_dim, out_dim, architecture, bool(args.use_bn)))


def build_width_command(args: argparse.Namespace, task_name: str, depth: int, root: Path) -> Tuple[str, ...]:
    return (
        sys.executable,
        repo_script("run_goliath_staged_width_only.py"),
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(args.results_dir),
        "--run-root",
        str(root),
        "--tasks",
        str(task_name),
        "--batch-size",
        str(int(args.batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--seed",
        str(int(args.seed)),
        "--stl-depth",
        str(int(depth)),
        "--alt-start-width",
        "1",
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
    )


def build_stl_command(args: argparse.Namespace, architecture: Sequence[int], root: Path) -> Tuple[str, ...]:
    return (
        sys.executable,
        repo_script("run_stl_ablation.py"),
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(args.results_dir),
        "--run-root",
        str(root),
        "--source-run-root",
        str(args.source_run_root),
        "--tasks",
        "anomaly",
        "--architecture",
        ",".join(str(int(v)) for v in architecture),
        "--repeat-count",
        str(int(args.repeat_count)),
        "--batch-size",
        str(int(args.batch_size)),
        "--no-pin-memory",
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
        "--metrics-every",
        "0",
        "--device",
        "auto",
    )


def build_jobs(args: argparse.Namespace) -> List[RecoveryJob]:
    root = Path(args.run_root)
    jobs: List[RecoveryJob] = []
    args.use_bn = True
    dims_cache: Dict[str, Tuple[int, int]] = {}
    if bool(args.include_width_only):
        depths = parse_csv_ints(args.width_depths)
        present_repeats = parse_csv_ints(args.missing_present_task_repeats)
        prediction_repeats = parse_csv_ints(args.prediction_repeats)
        for task_name in WIDTH_ONLY_PRESENT_TASKS:
            for repeat in present_repeats:
                for depth in depths:
                    job_root = root / "width_only" / task_name / f"repeat_{repeat:02d}" / f"d{depth:02d}"
                    jobs.append(
                        RecoveryJob(
                            name=f"width_only:{task_name}:r{repeat:02d}:d{depth:02d}",
                            kind="width_only",
                            task=task_name,
                            root=job_root,
                            command=build_width_command(args, task_name, depth, job_root),
                            parameter_count=estimated_width_params(task_name, depth, args, dims_cache),
                            depth=int(depth),
                        )
                    )
        for repeat in prediction_repeats:
            for depth in depths:
                job_root = root / "width_only" / "prediction" / f"repeat_{repeat:02d}" / f"d{depth:02d}"
                jobs.append(
                    RecoveryJob(
                        name=f"width_only:prediction:r{repeat:02d}:d{depth:02d}",
                        kind="width_only",
                        task="prediction",
                        root=job_root,
                        command=build_width_command(args, "prediction", depth, job_root),
                        parameter_count=estimated_width_params("prediction", depth, args, dims_cache),
                        depth=int(depth),
                    )
                )
    if bool(args.include_small_stl):
        for depth, width in MISSING_ANOMALY_STL_LEAVES:
            architecture = [int(width)] * int(depth)
            phase_name = stl.phase_name_for_architecture(architecture, 1)
            job_root = root / "small_grid_anomaly" / "_children" / phase_name
            jobs.append(
                RecoveryJob(
                    name=f"small_stl:anomaly:d{depth:02d}:w{width:03d}",
                    kind="small_stl",
                    task="anomaly",
                    root=job_root,
                    command=build_stl_command(args, architecture, job_root),
                    parameter_count=stl_params("anomaly", architecture, args, dims_cache),
                    depth=int(depth),
                )
            )
    return sorted(jobs, key=lambda job: (0 if job_state_path(job.root).exists() and not job_completed(job) else 1, job.parameter_count, job.depth, job.name))


def active_limit(args: argparse.Namespace, total_jobs: int) -> int:
    limit = int(args.max_active_jobs or 0)
    return max(1, total_jobs if limit <= 0 else limit)


def env_for(device_mode: str, gpu_index: int) -> Dict[str, str]:
    env = os.environ.copy()
    if device_mode == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device_mode == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
    return env


def choose_device(args: argparse.Namespace) -> Optional[str]:
    host = pressure.sample_host_memory_pressure()
    gpu = pressure.sample_gpu_memory_pressure(int(args.gpu_device_index))
    if host.used_pct > float(args.host_ram_resume_pct):
        return None
    if torch.cuda.is_available() and gpu.total_mib > 0 and gpu.used_pct <= float(args.gpu_memory_resume_pct):
        return "cuda"
    return "cpu"


def command_for_device(job: RecoveryJob, device_mode: str) -> List[str]:
    cmd = list(job.command)
    if job.kind != "small_stl":
        return cmd
    out: List[str] = []
    prev = ""
    for part in cmd:
        if part == "auto" and prev == "--device":
            out.append("cuda" if device_mode == "cuda" else "cpu")
        else:
            out.append(part)
        prev = part
    return out


def launch(job: RecoveryJob, device_mode: str, gpu_index: int, concurrency_hint: int, slot_index: int) -> Tuple[subprocess.Popen[Any], Any]:
    job.root.mkdir(parents=True, exist_ok=True)
    log_path = job_log_path(job.root)
    log_handle = log_path.open("a", encoding="utf-8")
    cmd = command_for_device(job, device_mode)
    env = launcher_child_env(
        env_for(device_mode, int(gpu_index)),
        concurrency_hint=concurrency_hint,
        job_key=f"{job.kind}:{job.task}:{job.root}",
        affinity_slot=slot_index,
    )
    proc = subprocess.Popen(cmd, start_new_session=True, env=env, stdout=log_handle, stderr=log_handle, text=True)
    return proc, log_handle


def terminate(proc: subprocess.Popen[Any], timeout_sec: float = 10.0) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout_sec)
        return
    except Exception:
        pass
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def write_plan(run_root: Path, jobs: Sequence[RecoveryJob]) -> None:
    rg.write_json(
        run_root / "recovery_plan.json",
        {
            "job_count": len(jobs),
            "jobs": [
                {
                    "name": job.name,
                    "kind": job.kind,
                    "task": job.task,
                    "root": str(job.root),
                    "parameter_count": int(job.parameter_count),
                    "depth": int(job.depth),
                    "command": list(job.command),
                }
                for job in jobs
            ],
        },
    )


def run(args: argparse.Namespace) -> None:
    os.environ["TABULAR_CPU_JOB_CONCURRENCY"] = str(int(detect_cpu_cores()))
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    logger = ContinuousLogger(run_root, "missing_width_stl_recovery", "pressure_aware_recovery", resume=(run_root / "training_log.txt").exists())
    jobs = build_jobs(args)
    write_plan(run_root, jobs)
    logger.log_console(f"Run root: {run_root}")
    logger.log_console(f"Jobs: {len(jobs)}")
    logger.log_console(f"Max active jobs: {active_limit(args, len(jobs))}")
    if bool(args.dry_run):
        for idx, job in enumerate(jobs[:50], start=1):
            logger.log_console(f"[PLAN] {idx:03d} {job.name} params={job.parameter_count} root={job.root}")
        logger.log_console("[DRY_RUN] plan written; no children launched")
        logger.close()
        return

    pending: Deque[RecoveryJob] = deque(jobs)
    active: Dict[subprocess.Popen[Any], ActiveRecoveryJob] = {}
    slot_count = max(1, min(int(detect_cpu_cores()), int(active_limit(args, len(jobs)))))
    free_slots = set(range(slot_count))
    launches_enabled = True
    limit = active_limit(args, len(jobs))
    while pending or active:
        finished: List[subprocess.Popen[Any]] = []
        for proc, active_job in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            finished.append(proc)
            try:
                if active_job.log_handle is not None:
                    active_job.log_handle.close()
            except Exception:
                pass
            job = active_job.job
            if job_completed(job):
                update_job_state(job.root, {"status": "completed", "completed": True, "exit_code": int(code or 0), "completed_at": time.time()})
                logger.log_console(f"[TASK] completed {job.name} params={job.parameter_count}")
                launches_enabled = True
                continue
            update_job_state(job.root, {"status": "paused" if active_job.pause_requested else "retrying", "completed": False, "exit_code": int(code or 0), "updated_at": time.time()})
            pending.appendleft(job)
            launches_enabled = False
            logger.log_console(f"[TASK] requeued {job.name} exit={code} pause={active_job.pause_requested}")

        for proc in finished:
            entry = active.pop(proc, None)
            if entry is not None:
                free_slots.add(entry.slot_index)

        host = pressure.sample_host_memory_pressure()
        gpu = pressure.sample_gpu_memory_pressure(int(args.gpu_device_index))
        if active and gpu.total_mib > 0 and gpu.used_pct > float(args.gpu_memory_pressure_limit_pct):
            gpu_active = [entry for entry in active.values() if entry.device_mode == "cuda" and not entry.pause_requested]
            if gpu_active:
                victim = max(gpu_active, key=lambda entry: (entry.job.parameter_count, entry.job.depth, entry.job.name))
                victim.pause_requested = True
                update_job_state(victim.job.root, {"status": "pausing", "reason": "gpu_memory_pressure", "completed": False})
                logger.log_console(f"[PRESSURE] pause_gpu {victim.job.name} gpu_used_pct={gpu.used_pct:.2f}")
                terminate(next(proc for proc, entry in active.items() if entry is victim))
                time.sleep(max(0.0, float(args.pressure_settle_sec)))
                continue
        if active and host.used_pct > float(args.host_ram_pressure_limit_pct):
            pausable = [entry for entry in active.values() if not entry.pause_requested]
            if len(active) > 1 and pausable:
                victim = max(pausable, key=lambda entry: (entry.job.parameter_count, entry.job.depth, entry.job.name))
                victim.pause_requested = True
                update_job_state(victim.job.root, {"status": "pausing", "reason": "host_ram_pressure", "completed": False})
                logger.log_console(f"[PRESSURE] pause_host {victim.job.name} host_used_pct={host.used_pct:.2f}")
                terminate(next(proc for proc, entry in active.items() if entry is victim))
                time.sleep(max(0.0, float(args.pressure_settle_sec)))
                continue

        device_mode = choose_device(args)
        if pending and len(active) < limit and free_slots and launches_enabled and (device_mode is not None or not active):
            job = pending.popleft()
            if job_completed(job):
                update_job_state(job.root, {"status": "completed", "completed": True, "completed_at": time.time()})
                logger.log_console(f"[TASK] already_complete {job.name}")
                continue
            chosen = device_mode or "cpu"
            slot_index = min(free_slots)
            free_slots.remove(slot_index)
            proc, handle = launch(job, chosen, int(args.gpu_device_index), concurrency_hint=len(active) + 1, slot_index=slot_index)
            active[proc] = ActiveRecoveryJob(job=job, device_mode=chosen, log_path=job_log_path(job.root), log_handle=handle, slot_index=slot_index)
            update_job_state(job.root, {"status": "running", "completed": False, "device": chosen, "started_at": time.time(), "command": command_for_device(job, chosen)})
            logger.log_console(
                f"[TASK] launch {job.name} params={job.parameter_count} device={chosen} "
                f"host_used_pct={host.used_pct:.2f} gpu_used_pct={gpu.used_pct:.2f} active={len(active)}/{limit}"
            )
            time.sleep(max(0.0, float(args.pressure_settle_sec)))
            continue
        time.sleep(max(0.1, float(args.pressure_poll_interval_sec)))
    logger.log_console("[DONE] all recovery jobs completed")
    logger.close()


def main() -> None:
    bootstrap_runtime("run_missing_width_stl_recovery_pressure")

    args = parse_args()
    if float(args.host_ram_resume_pct) > float(args.host_ram_pressure_limit_pct):
        raise SystemExit("--host-ram-resume-pct must be <= --host-ram-pressure-limit-pct")
    if float(args.gpu_memory_resume_pct) > float(args.gpu_memory_pressure_limit_pct):
        raise SystemExit("--gpu-memory-resume-pct must be <= --gpu-memory-pressure-limit-pct")
    run(args)


if __name__ == "__main__":
    main()
