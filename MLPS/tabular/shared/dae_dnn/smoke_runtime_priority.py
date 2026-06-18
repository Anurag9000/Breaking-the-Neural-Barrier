from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from MLPS.tabular.shared.dae_dnn.platform_runtime import popen_process_group_kwargs
from MLPS.tabular.shared.dae_dnn.runtime_tuning import (
    bootstrap_runtime,
    detect_cpu_cores,
    launcher_child_env,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test tabular runtime priority and CPU saturation.")
    p.add_argument("--child", action="store_true", default=False)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--sample-interval", type=float, default=0.5)
    p.add_argument("--min-average-cpu-pct", type=float, default=85.0)
    return p.parse_args()


def _read_proc_stat() -> Tuple[int, int]:
    with open("/proc/stat", "r", encoding="utf-8") as f:
        first = f.readline().strip().split()
    if not first or first[0] != "cpu":
        raise RuntimeError("Unable to read /proc/stat")
    values = [int(v) for v in first[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def _cpu_utilization_pct(interval_sec: float) -> float:
    total_a, idle_a = _read_proc_stat()
    time.sleep(max(0.05, float(interval_sec)))
    total_b, idle_b = _read_proc_stat()
    total_delta = max(1, total_b - total_a)
    idle_delta = max(0, idle_b - idle_a)
    busy_delta = max(0, total_delta - idle_delta)
    return 100.0 * (busy_delta / float(total_delta))


def _scheduler_name(policy: int) -> str:
    if policy == getattr(os, "SCHED_BATCH", -999):
        return "SCHED_BATCH"
    if policy == getattr(os, "SCHED_OTHER", -999):
        return "SCHED_OTHER"
    if policy == getattr(os, "SCHED_FIFO", -999):
        return "SCHED_FIFO"
    if policy == getattr(os, "SCHED_RR", -999):
        return "SCHED_RR"
    return str(policy)


def _busy_loop(seconds: float) -> None:
    deadline = time.monotonic() + max(0.1, float(seconds))
    x = 0
    while time.monotonic() < deadline:
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
    if x == -1:
        print("unreachable", flush=True)


def run_child(args: argparse.Namespace) -> int:
    if os.name != "posix" or not Path("/proc/self/cgroup").exists():
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "runtime priority smoke child requires POSIX /proc and scheduler APIs",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    info = bootstrap_runtime("smoke_runtime_priority_child")
    payload = {
        "pid": os.getpid(),
        "cgroup": Path("/proc/self/cgroup").read_text(encoding="utf-8").strip(),
        "scheduler": _scheduler_name(os.sched_getscheduler(0)),
        "affinity": sorted(int(cpu) for cpu in os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "threads": int(info["torch_threads"]),
        "workers": int(info["num_workers"]),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    _busy_loop(args.seconds)
    return 0


def _read_child_payloads(processes: Sequence[subprocess.Popen[str]]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for proc in processes:
        if proc.stdout is None:
            raise RuntimeError("Missing child stdout pipe")
        line = proc.stdout.readline().strip()
        if not line:
            raise RuntimeError(f"Child {proc.pid} did not emit bootstrap payload")
        payloads.append(json.loads(line))
    return payloads


def _check_affinity_disjoint(payloads: Sequence[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    sets = [(payload["pid"], set(int(cpu) for cpu in payload["affinity"])) for payload in payloads]
    for idx, (pid_a, cpus_a) in enumerate(sets):
        for pid_b, cpus_b in sets[idx + 1 :]:
            overlap = sorted(cpus_a & cpus_b)
            if overlap:
                problems.append(f"pid {pid_a} overlaps pid {pid_b} on CPUs {overlap}")
    return not problems, problems


def run_parent(args: argparse.Namespace) -> int:
    if os.name != "posix" or not Path("/proc/stat").exists():
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "runtime priority smoke test requires POSIX /proc and scheduler APIs",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    parent_info = bootstrap_runtime("smoke_runtime_priority_parent")
    workers = int(args.workers) if int(args.workers) > 0 else int(detect_cpu_cores())
    workers = max(1, workers)

    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--seconds", str(float(args.seconds))]
    processes: List[subprocess.Popen[str]] = []
    try:
        for slot in range(workers):
            env = launcher_child_env(
                concurrency_hint=workers,
                job_key=f"smoke-slot:{slot}",
                affinity_slot=slot,
            )
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **popen_process_group_kwargs(),
            )
            processes.append(proc)

        payloads = _read_child_payloads(processes)
        utilizations: List[float] = []
        deadline = time.monotonic() + max(0.5, float(args.seconds))
        while time.monotonic() < deadline:
            utilizations.append(_cpu_utilization_pct(args.sample_interval))
        average_cpu = sum(utilizations) / float(len(utilizations) or 1)
        peak_cpu = max(utilizations) if utilizations else 0.0
    finally:
        for proc in processes:
            try:
                proc.wait(timeout=max(1.0, float(args.seconds) + 2.0))
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    child_failures = [proc for proc in processes if proc.returncode not in (0, None)]
    disjoint, overlap_problems = _check_affinity_disjoint(payloads)
    scope_ok = all("app-mlps-training.slice" in str(payload["cgroup"]) for payload in payloads)
    sched_ok = all(payload["scheduler"] == "SCHED_BATCH" for payload in payloads)

    report = {
        "parent": parent_info,
        "workers": workers,
        "average_cpu_pct": round(average_cpu, 2),
        "peak_cpu_pct": round(peak_cpu, 2),
        "scope_ok": scope_ok,
        "sched_ok": sched_ok,
        "affinity_disjoint": disjoint,
        "child_failures": [proc.pid for proc in child_failures],
        "child_payloads": payloads,
        "affinity_overlap_problems": overlap_problems,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    if child_failures:
        return 1
    if not scope_ok or not sched_ok or not disjoint:
        return 2
    if average_cpu < float(args.min_average_cpu_pct):
        return 3
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run_child(args) if bool(args.child) else run_parent(args))


if __name__ == "__main__":
    main()
