from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import time
from typing import Any, Dict, Optional, Tuple


def sample_host_memory_mib() -> Tuple[int, int]:
    """Return (total_mib, available_mib) on Linux, Windows, or best effort."""

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            total_mib = 0
            available_mib = 0
            for line in handle:
                if line.startswith("MemTotal:"):
                    total_mib = int(int(line.split()[1]) // 1024)
                elif line.startswith("MemAvailable:"):
                    available_mib = int(int(line.split()[1]) // 1024)
                if total_mib and available_mib:
                    return int(total_mib), int(available_mib)
    except Exception:
        pass

    if os.name == "nt":
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
                return int(status.ullTotalPhys // (1024 * 1024)), int(status.ullAvailPhys // (1024 * 1024))
        except Exception:
            pass

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total // (1024 * 1024)), int(vm.available // (1024 * 1024))
    except Exception:
        return 1, 0


def popen_process_group_kwargs() -> Dict[str, Any]:
    if os.name == "nt":
        flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flag} if flag else {}
    return {"start_new_session": True}


def terminate_process_tree(proc: subprocess.Popen[Any], timeout_sec: float = 10.0) -> None:
    """Terminate a launcher child and its descendants on POSIX or Windows."""

    if os.name == "nt":
        try:
            proc.send_signal(getattr(signal, "CTRL_BREAK_EVENT"))
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=timeout_sec)
            return
        except Exception:
            pass
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=timeout_sec)
        except Exception:
            pass
        return

    try:
        pgid: Optional[int] = os.getpgid(proc.pid)
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
    deadline = time.time() + max(0.1, float(timeout_sec))
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
