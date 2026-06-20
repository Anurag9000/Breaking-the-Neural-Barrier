"""Best-effort runtime tuning for tabular DAE/DNN launchers.

This module centralizes process-level tuning so the training runners can
inherit a high-throughput CPU configuration without each script reimplementing
the same boilerplate. It is intentionally best effort: permission failures for
priority changes are ignored.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
from typing import Dict, Optional, Sequence, Tuple

if os.name == "nt":
    from ctypes import wintypes

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


def _env_truthy(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on", "y"}:
        return True
    if lowered in {"0", "false", "no", "off", "n", ""}:
        return False
    return default


def strict_no_swap_enabled() -> bool:
    return _env_truthy(os.environ.get("TABULAR_STRICT_NO_SWAP"), default=True)


def _current_affinity_cpus() -> Tuple[int, ...]:
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return _order_cpus_by_capacity(tuple(int(cpu) for cpu in affinity))
    except Exception:
        pass
    return _order_cpus_by_capacity(tuple(range(max(1, int(os.cpu_count() or 1)))))


def _cpu_capacity_khz(cpu: int) -> int:
    cpu_dir = f"/sys/devices/system/cpu/cpu{int(cpu)}/cpufreq"
    for name in ("cpuinfo_max_freq", "scaling_max_freq", "base_frequency"):
        path = os.path.join(cpu_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return int(handle.read().strip() or "0")
        except Exception:
            pass
    return 0


def _order_cpus_by_capacity(cpus: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted((int(cpu) for cpu in cpus), key=lambda cpu: (-_cpu_capacity_khz(cpu), int(cpu))))


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
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            process = kernel32.GetCurrentProcess()
            cpus = set(_parse_cpu_list(affinity_text)) if affinity_text else set()
            if not cpus:
                cpus = set(range(max(1, int(os.cpu_count() or 1))))
            mask = 0
            for cpu in cpus:
                if 0 <= int(cpu) < 64:
                    mask |= 1 << int(cpu)
            if mask:
                kernel32.SetProcessAffinityMask(process, mask)
        except Exception:
            pass
        return

    if affinity_text:
        try:
            cpus = set(_parse_cpu_list(affinity_text))
            if cpus:
                os.sched_setaffinity(0, cpus)
                return
        except Exception:
            pass
    try:
        all_online = set(range(max(1, int(os.cpu_count() or 1))))
        if all_online:
            os.sched_setaffinity(0, all_online)
    except Exception:
        pass


def current_concurrency_hint(default: Optional[int] = None) -> Optional[int]:
    explicit = _safe_int(str(default)) if default is not None else None
    if explicit is not None:
        return explicit
    hint = _safe_int(os.environ.get("TABULAR_CPU_JOB_CONCURRENCY"))
    if hint is not None:
        return hint
    return None


def derive_cpu_budget(concurrency_hint: Optional[int] = None) -> Tuple[int, int, int]:
    """Return (thread_budget, worker_budget, detected_cores).

    The per-process thread budget is the detected core count divided by the
    active launcher concurrency hint. When there is no concurrency hint, a
    single process can use the whole machine. Worker count is intentionally
    aggressive so the default path keeps data loading and preprocessing busy.
    """

    cores = detect_cpu_cores()
    hint = current_concurrency_hint(concurrency_hint)
    if hint is None or hint <= 1:
        thread_budget = cores
        worker_budget = cores
        return thread_budget, worker_budget, cores

    thread_budget = max(1, cores // int(hint))
    worker_budget = max(1, thread_budget)
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
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            process = kernel32.GetCurrentProcess()
            priority_name = str(os.environ.get("TABULAR_WINDOWS_PRIORITY_CLASS", "high")).strip().lower()
            priority_map = {
                "idle": 0x00000040,  # IDLE_PRIORITY_CLASS
                "below_normal": 0x00004000,  # BELOW_NORMAL_PRIORITY_CLASS
                "normal": 0x00000020,  # NORMAL_PRIORITY_CLASS
                "above_normal": 0x00008000,  # ABOVE_NORMAL_PRIORITY_CLASS
                "high": 0x00000080,  # HIGH_PRIORITY_CLASS
                "background": 0x00100000,  # PROCESS_MODE_BACKGROUND_BEGIN
            }
            priority_class = priority_map.get(priority_name, priority_map["high"])
            kernel32.SetPriorityClass(process, priority_class)
        except Exception:
            pass
        try:
            kernel32 = ctypes.windll.kernel32
            process = kernel32.GetCurrentProcess()
            memory_priority_name = str(os.environ.get("TABULAR_WINDOWS_MEMORY_PRIORITY", "normal")).strip().lower()
            memory_priority_map = {
                "very_low": 1,
                "low": 2,
                "medium": 3,
                "below_normal": 4,
                "normal": 5,
            }
            memory_priority = memory_priority_map.get(memory_priority_name, 5)
            class MEMORY_PRIORITY_INFORMATION(ctypes.Structure):
                _fields_ = [("MemoryPriority", wintypes.ULONG)]

            info = MEMORY_PRIORITY_INFORMATION(memory_priority)
            # PROCESS_INFORMATION_CLASS::ProcessMemoryPriority = 0
            kernel32.SetProcessInformation(process, 0, ctypes.byref(info), ctypes.sizeof(info))
        except Exception:
            pass
        return

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

    sched_batch = getattr(os, "SCHED_BATCH", None)
    if sched_batch is not None:
        try:
            os.sched_setscheduler(0, sched_batch, os.sched_param(0))
        except Exception:
            pass


def _linux_active_swap_entries() -> Tuple[str, ...]:
    try:
        with open("/proc/swaps", "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines()[1:] if line.strip()]
    except Exception:
        return tuple()
    entries = []
    for line in lines:
        parts = line.split()
        if parts:
            entries.append(parts[0])
    return tuple(entries)


def _linux_swap_used_mib() -> int:
    total_kib = 0
    free_kib = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("SwapTotal:"):
                    total_kib = int(line.split()[1])
                elif line.startswith("SwapFree:"):
                    free_kib = int(line.split()[1])
                if total_kib and free_kib >= 0:
                    break
    except Exception:
        return 0
    if total_kib <= 0:
        return 0
    return max(0, int((total_kib - free_kib) // 1024))


def _current_cgroup_dir() -> Optional[str]:
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(":", 2)
                if len(parts) == 3:
                    rel = parts[2].strip()
                    if rel:
                        return rel.lstrip("/")
    except Exception:
        return None
    return None


def _read_cgroup_value(filename: str) -> Optional[str]:
    rel = _current_cgroup_dir()
    candidates = []
    if rel is not None:
        candidates.append(os.path.join("/sys/fs/cgroup", rel, filename))
    candidates.append(os.path.join("/sys/fs/cgroup", filename))
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except Exception:
            continue
    return None


def _windows_pagefile_enabled() -> bool:
    try:
        import psutil  # type: ignore

        swap = psutil.swap_memory()
        if int(getattr(swap, "total", 0) or 0) > 0:
            return True
    except Exception:
        pass

    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            slack = 16 * 1024 * 1024
            return int(status.ullTotalPageFile) > int(status.ullTotalPhys) + int(slack)
    except Exception:
        pass
    return False


def enforce_strict_no_swap(label: str) -> None:
    if not strict_no_swap_enabled():
        return

    if os.name == "nt":
        if _windows_pagefile_enabled():
            raise SystemExit(
                f"{label}: strict no-swap mode is enabled, but Windows pagefile usage is available. "
                "Disable the pagefile system-wide before launching this repo."
            )
        return

    active_swaps = _linux_active_swap_entries()
    if active_swaps:
        raise SystemExit(
            f"{label}: strict no-swap mode is enabled, but active Linux swap devices exist: {', '.join(active_swaps)}. "
            "Disable swap system-wide, for example with `sudo swapoff -a`, before launching this repo."
        )

    if _linux_swap_used_mib() > 0:
        raise SystemExit(f"{label}: strict no-swap mode is enabled, but host swap usage is already non-zero.")

    swap_max = _read_cgroup_value("memory.swap.max")
    if os.environ.get("TABULAR_SYSTEMD_SCOPED") == "1" and swap_max not in {None, "", "0"}:
        raise SystemExit(
            f"{label}: strict no-swap mode expected memory.swap.max=0 inside the launcher scope, got {swap_max!r}."
        )

    zswap_max = _read_cgroup_value("memory.zswap.max")
    if os.environ.get("TABULAR_SYSTEMD_SCOPED") == "1" and zswap_max not in {None, "", "0"}:
        raise SystemExit(
            f"{label}: strict no-swap mode expected memory.zswap.max=0 inside the launcher scope, got {zswap_max!r}."
        )


def _scope_properties() -> Tuple[str, ...]:
    props = [
        "CPUWeight=10000",
        "StartupCPUWeight=10000",
        "IOWeight=10000",
        "StartupIOWeight=10000",
        "MemorySwapMax=0",
        "TasksMax=infinity",
        "CPUAccounting=yes",
        "IOAccounting=yes",
        "MemoryAccounting=yes",
        "TasksAccounting=yes",
    ]
    try:
        systemctl = shutil.which("systemctl")
        if systemctl is not None:
            proc = subprocess.run(
                [systemctl, "--version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            first_line = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
            parts = first_line.split()
            if len(parts) >= 2 and parts[0].lower() == "systemd":
                version = int(parts[1])
                if version >= 253:
                    props.append("MemoryZSwapMax=0")
    except Exception:
        pass
    affinity_cpus = _current_affinity_cpus()
    if affinity_cpus:
        props.append(f"AllowedCPUs={_format_cpu_list(affinity_cpus)}")
    return tuple(props)


def _maybe_reexec_under_systemd_scope(label: str) -> None:
    if os.name != "posix":
        return
    if os.environ.get("TABULAR_SKIP_SYSTEMD_SCOPE") == "1":
        return
    if os.environ.get("TABULAR_SYSTEMD_SCOPED") == "1":
        return
    if sys.argv[:1] and sys.argv[0] in {"-c", "-"}:
        return
    if shutil.which("systemd-run") is None:
        return

    scope_cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--same-dir",
        "--collect",
        "--slice=app-mlps-training.slice",
        f"--description=tabular:{label}",
    ]
    for prop in _scope_properties():
        scope_cmd.extend(["-p", prop])
    scope_cmd.extend([sys.executable, *sys.argv])

    env = dict(os.environ)
    env["TABULAR_SYSTEMD_SCOPED"] = "1"
    try:
        os.execvpe(scope_cmd[0], scope_cmd, env)
    except Exception:
        return


def launcher_child_env(
    base_env: Optional[Dict[str, str]] = None,
    *,
    concurrency_hint: Optional[int] = None,
    job_key: Optional[str] = None,
    affinity_slot: Optional[int] = None,
    shared_cpu: bool = False,
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
    # Shared CPU mode is the default: children inherit the full visible CPU set
    # unless a caller explicitly disables it.
    shared_cpu = bool(shared_cpu) or _env_truthy(env.get("TABULAR_CHILD_SHARED_CPU"), default=True)
    if shared_cpu:
        thread_budget = max(1, int(cores))
        worker_budget = max(1, int(cores))
    elif affinity_slot is not None and hint and len(affinity_cpus) > 1:
        slot_count = min(max(1, int(hint)), len(affinity_cpus))
        slot = max(0, min(int(affinity_slot), slot_count - 1))
        affinity_cpus = _partition_cpus(affinity_cpus, slot_count, slot)
    elif job_key and hint and len(affinity_cpus) > 1:
        slot_count = min(max(1, int(hint)), len(affinity_cpus))
        slot = _deterministic_slot(job_key, slot_count)
        affinity_cpus = _partition_cpus(affinity_cpus, slot_count, slot)
    if affinity_cpus:
        affinity_text = _format_cpu_list(affinity_cpus)
        env["TABULAR_CPU_AFFINITY_CPUS"] = affinity_text
        if shared_cpu:
            env["GOMP_CPU_AFFINITY"] = affinity_text
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        env[key] = str(thread_budget)
    env["TORCH_INTEROP_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["OMP_WAIT_POLICY"] = "ACTIVE"
    if shared_cpu:
        env["OMP_PROC_BIND"] = "spread"
        env["OMP_PLACES"] = "cores"
        env["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    env["TABULAR_CPU_THREADS"] = str(thread_budget)
    env["TABULAR_CPU_WORKERS"] = str(worker_budget)
    env["TABULAR_CPU_CORES"] = str(cores)
    env["TABULAR_CPU_JOB_CONCURRENCY"] = "1" if shared_cpu else str(max(1, int(current_concurrency_hint(concurrency_hint) or 1)))
    return env


def bootstrap_runtime(label: str = "tabular") -> Dict[str, int]:
    """Apply best-effort runtime tuning and return the selected settings."""

    _maybe_reexec_under_systemd_scope(label)
    enforce_strict_no_swap(label)
    _apply_affinity_from_env()
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
        "TABULAR_CHILD_SHARED_CPU": "1",
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
