"""Best-effort runtime tuning for tabular DAE/DNN launchers.

This module centralizes process-level tuning so the training runners can
inherit a high-throughput CPU configuration without each script reimplementing
the same boilerplate. It is intentionally best effort: permission failures for
priority changes are ignored.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Optional, Tuple

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


def launcher_child_env(base_env: Optional[Dict[str, str]] = None, *, concurrency_hint: Optional[int] = None) -> Dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    thread_budget, worker_budget, cores = derive_cpu_budget(concurrency_hint)
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
        "NUMEXPR_NUM_THREADS": str(thread_budget),
        "VECLIB_MAXIMUM_THREADS": str(thread_budget),
        "TORCH_NUM_THREADS": str(thread_budget),
        "TORCH_INTEROP_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "TABULAR_CPU_THREADS": str(thread_budget),
        "TABULAR_CPU_WORKERS": str(worker_budget),
        "TABULAR_CPU_CORES": str(cpu_cores),
    }
    for key, value in env_updates.items():
        os.environ[key] = value

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
