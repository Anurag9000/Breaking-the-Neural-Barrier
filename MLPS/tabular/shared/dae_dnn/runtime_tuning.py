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
from typing import Dict

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


def resolve_num_workers(requested: int | None = None) -> int:
    """Resolve DataLoader worker count.

    A positive explicit request wins. Otherwise we default to the full logical
    CPU count so the data pipeline can use the machine aggressively when the
    launcher leaves workers unspecified or zero.
    """

    if requested is not None:
        requested = int(requested)
        if requested > 0:
            return requested

    env_override = os.environ.get("TABULAR_CPU_WORKERS")
    if env_override:
        try:
            env_workers = int(env_override)
        except Exception:
            env_workers = 0
        if env_workers > 0:
            return env_workers

    return detect_cpu_cores()


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


def bootstrap_runtime(label: str = "tabular") -> Dict[str, int]:
    """Apply best-effort runtime tuning and return the selected settings."""

    cpu_cores = detect_cpu_cores()
    env_updates = {
        "OMP_NUM_THREADS": str(cpu_cores),
        "MKL_NUM_THREADS": str(cpu_cores),
        "OPENBLAS_NUM_THREADS": str(cpu_cores),
        "NUMEXPR_NUM_THREADS": str(cpu_cores),
        "VECLIB_MAXIMUM_THREADS": str(cpu_cores),
        "TORCH_NUM_THREADS": str(cpu_cores),
        "TORCH_INTEROP_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
    }
    for key, value in env_updates.items():
        os.environ[key] = value

    _apply_process_priority()

    if torch is not None:
        try:
            torch.set_num_threads(cpu_cores)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    return {
        "label": label,
        "cpu_cores": cpu_cores,
        "num_workers": resolve_num_workers(None),
        "torch_threads": cpu_cores,
        "torch_interop_threads": 1,
    }
