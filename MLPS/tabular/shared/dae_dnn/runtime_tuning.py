"""Best-effort runtime tuning for tabular DAE/DNN launchers.

This module centralizes process-level tuning so the training runners can
inherit a high-throughput CPU configuration without each script reimplementing
the same boilerplate. It is intentionally best effort: permission failures for
priority changes are ignored.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from typing import Dict, Optional, Sequence, Tuple

try:
    import torch
except Exception:  # pragma: no cover - torch is always available in normal runs
    torch = None


def detect_cpu_cores() -> int:
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return max(1, len(affinity))
    except Exception:
        pass
    return max(1, int(os.cpu_count() or 1))


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _current_affinity_cpus() -> Tuple[int, ...]:
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return tuple(sorted(int(cpu) for cpu in affinity))
    except Exception:
        pass
    return tuple(range(max(1, int(os.cpu_count() or 1))))


def _format_cpu_list(cpus: Sequence[int]) -> str:
    return ",".join(str(int(cpu)) for cpu in cpus)


def _partition_cpus(cpus: Sequence[int], parts: int, slot: int) -> Tuple[int, ...]:
    cpus = tuple(int(cpu) for cpu in cpus)
    if not cpus:
        return tuple()
    parts = max(1, min(int(parts), len(cpus)))
    slot = max(0, min(int(slot), parts - 1))
    base, remainder = divmod(len(cpus), parts)
    start = 0
    for index in range(parts):
        size = base + (1 if index < remainder else 0)
        if index == slot:
            return cpus[start : start + size]
        start += size
    return cpus


def _deterministic_slot(key: str, parts: int) -> int:
    if parts <= 1:
        return 0
    digest = hashlib.sha1(str(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(parts)


def _parse_cpu_list(text: str) -> Tuple[int, ...]:
    cpus = set()
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                start, end = end, start
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    return tuple(sorted(cpus))


def _apply_affinity_from_env() -> None:
    affinity_text = os.environ.get("TABULAR_CPU_AFFINITY_CPUS")
    if affinity_text:
        try:
            cpus = set(_parse_cpu_list(affinity_text))
            if cpus:
                os.sched_setaffinity(0, cpus)
                return
        except Exception:
            pass
    try:
        current = os.sched_getaffinity(0)
        if current:
            os.sched_setaffinity(0, current)
    except Exception:
        pass


def current_concurrency_hint(default: Optional[int] = None) -> Optional[int]:
    hint = _safe_int(os.environ.get("TABULAR_CPU_JOB_CONCURRENCY"))
    if hint is not None:
        return hint
    return default


def derive_cpu_budget(concurrency_hint: Optional[int] = None) -> Tuple[int, int, int]:
    """Return (thread_budget, worker_budget, detected_cores).

    The per-process thread budget is the detected core count divided by the
    active launcher concurrency hint. When there is no concurrency hint, a
    single process can use the whole machine. Worker count stays deliberately
    small to avoid process explosion under many concurrent children.
    """

    cores = detect_cpu_cores()
    hint = current_concurrency_hint(concurrency_hint)
    if hint is None or hint <= 1:
        thread_budget = cores
        worker_budget = max(1, min(4, cores // 4))
        return thread_budget, worker_budget, cores

    thread_budget = max(1, cores // int(hint))
    worker_budget = 1 if thread_budget >= 1 else 0
    return thread_budget, worker_budget, cores


def resolve_num_workers(requested: int | None = None) -> int:
    """Resolve DataLoader worker count.

    A positive explicit request wins. Otherwise an environment override wins.
    If neither is provided, the count is derived from the current concurrency
    budget rather than blindly claiming every core.
    """

    if requested is not None:
        requested = int(requested)
        if requested > 0:
            return requested

    env_override = _safe_int(os.environ.get("TABULAR_CPU_WORKERS"))
    if env_override is not None:
        return env_override

    _, worker_budget, _ = derive_cpu_budget()
    return max(0, int(worker_budget))


def _apply_process_priority() -> None:
    try:
        os.nice(-20)
    except Exception:
        pass

    if shutil.which("ionice") is not None:
        try:
            subprocess.run(
                ["ionice", "-c2", "-n0", "-p", str(os.getpid())],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def launcher_child_env(
    base_env: Optional[Dict[str, str]] = None,
    *,
    concurrency_hint: Optional[int] = None,
    job_key: Optional[str] = None,
) -> Dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    thread_budget, worker_budget, cores = derive_cpu_budget(concurrency_hint)
    hint = current_concurrency_hint(concurrency_hint)
    affinity_cpus = _current_affinity_cpus()
    affinity_source = env.get("TABULAR_CPU_AFFINITY_CPUS") or os.environ.get("TABULAR_CPU_AFFINITY_CPUS")
    if affinity_source:
        try:
            parsed = _parse_cpu_list(affinity_source)
            if parsed:
                affinity_cpus = parsed
        except Exception:
            pass
    if job_key and hint and len(affinity_cpus) > 1:
        slot_count = min(max(1, int(hint)), len(affinity_cpus))
        slot = _deterministic_slot(job_key, slot_count)
        affinity_cpus = _partition_cpus(affinity_cpus, slot_count, slot)
    if affinity_cpus:
        env["TABULAR_CPU_AFFINITY_CPUS"] = _format_cpu_list(affinity_cpus)
    env["TABULAR_CPU_THREADS"] = str(thread_budget)
    env["TABULAR_CPU_WORKERS"] = str(worker_budget)
    env["TABULAR_CPU_CORES"] = str(cores)
    env["TABULAR_CPU_JOB_CONCURRENCY"] = str(max(1, int(current_concurrency_hint(concurrency_hint) or 1)))
    return env


def bootstrap_runtime(label: str = "tabular") -> Dict[str, int]:
    """Apply best-effort runtime tuning and return the selected settings."""

    thread_budget, worker_budget, cpu_cores = derive_cpu_budget()
    env_updates = {
        "OMP_NUM_THREADS": str(thread_budget),
        "MKL_NUM_THREADS": str(thread_budget),
        "OPENBLAS_NUM_THREADS": str(thread_budget),
        "GOTO_NUM_THREADS": str(thread_budget),
        "NUMEXPR_NUM_THREADS": str(thread_budget),
        "VECLIB_MAXIMUM_THREADS": str(thread_budget),
        "TORCH_NUM_THREADS": str(thread_budget),
        "TORCH_INTEROP_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "OMP_WAIT_POLICY": "ACTIVE",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "cores",
        "KMP_AFFINITY": "granularity=fine,scatter",
        "TABULAR_CPU_THREADS": str(thread_budget),
        "TABULAR_CPU_WORKERS": str(worker_budget),
        "TABULAR_CPU_CORES": str(cpu_cores),
    }
    for key, value in env_updates.items():
        os.environ[key] = value

    _apply_affinity_from_env()
    _apply_process_priority()

    if torch is not None:
        try:
            torch.set_num_threads(thread_budget)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    return {
        "label": label,
        "cpu_cores": cpu_cores,
        "num_workers": worker_budget,
        "torch_threads": thread_budget,
        "torch_interop_threads": 1,
    }
