from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from MLPS.tabular.shared.dae_dnn.mlp import MLP
from MLPS.tabular.shared.dae_dnn.tasks import build_task, refresh_task_loaders
from MLPS.tabular.shared.dae_dnn.runtime_tuning import bootstrap_runtime
from MLPS.tabular.shared.dae_dnn.train_utils import unpack_batch


DEFAULT_TASKS = [
    "classification",
    "simulation",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe the largest width per depth that stays under a VRAM threshold for two training batches.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default=None)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--batch-size", type=int, default=0, help="Batch size override. 0 (default) defers to per-task target-batches computation.")
    p.add_argument(
        "--task-batch-size",
        action="append",
        default=[],
        help="Override batch size for a specific task, e.g. classification=93120. May be passed multiple times.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-depth", type=int, default=1)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--min-width", type=int, default=16)
    p.add_argument("--width-step", type=int, default=16)
    p.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Maximum width to test. Use 0 or a negative value for no explicit ceiling.",
    )
    p.add_argument("--calibrated-limit-mib", type=int, default=9472)
    p.add_argument("--search-margin", type=int, default=128)
    p.add_argument("--vram-threshold-mib", type=int, default=6144)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-bn", action="store_true", default=True)
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    p.add_argument("--clear-results", action="store_true", default=False)
    return p.parse_args()


def parse_task_batch_sizes(values: Sequence[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for value in values:
        text = str(value).strip()
        if not text or "=" not in text:
            continue
        task_name, batch_size_text = text.split("=", 1)
        task_name = task_name.strip().lower()
        try:
            batch_size = int(batch_size_text.strip())
        except Exception:
            continue
        if task_name:
            mapping[task_name] = max(1, batch_size)
    return mapping


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def query_gpu_memory_used_mib(device_index: int = 0) -> Optional[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={int(device_index)}",
                "--query-gpu=memory.used",
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


@dataclass
class ProbeResult:
    task: str
    depth: int
    width: int
    success: bool
    reason: str
    batches_completed: int
    epochs_completed: int
    peak_mib: int
    peak_allocated_mib: int
    peak_reserved_mib: int


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    gc.collect()


def terminate_stale_processes(exclude_pids: Optional[Sequence[int]] = None) -> None:
    exclude = {int(pid) for pid in (exclude_pids or [])}
    patterns = [
        "MLPS/tabular/shared/dae_dnn/probe_width_capacity.py",
        "MLPS/tabular/shared/dae_dnn/run_stl_ablation.py",
        "MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py",
        "MLPS/tabular/shared/dae_dnn/run_with_watchdog.py",
    ]
    pids_to_kill = set()
    for pattern in patterns:
        try:
            out = subprocess.check_output(["pgrep", "-af", pattern], stderr=subprocess.DEVNULL).decode("utf-8")
        except Exception:
            continue
        for line in out.splitlines():
            parts = line.strip().split(None, 1)
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except Exception:
                continue
            if pid in exclude:
                continue
            pids_to_kill.add(pid)

    for pid in sorted(pids_to_kill):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    if pids_to_kill:
        time.sleep(1.0)
    for pid in sorted(pids_to_kill):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def align_width(width: int, min_width: int, step: int) -> int:
    width = max(int(min_width), int(width))
    if step > 0:
        width = int(width // int(step) * int(step))
        width = max(int(min_width), width)
    return width


def binary_search_max_width(
    *,
    probe,
    start_width: int,
    min_width: int,
    max_width: Optional[int],
    width_step: int,
) -> tuple[Optional[int], Optional[int], Optional[str], Optional[int], Optional[int], Optional[int]]:
    """
    Return the highest passing width and the first failing width using
    exponential bracketing followed by binary search.
    """
    min_width = int(min_width)
    max_width = None if max_width is None or int(max_width) <= 0 else int(max_width)
    width_step = max(1, int(width_step))
    start_width = align_width(int(start_width), min_width, width_step)
    if max_width is not None:
        start_width = min(start_width, max_width)

    best_width: Optional[int] = None
    best_peak: Optional[int] = None
    best_allocated: Optional[int] = None
    best_reserved: Optional[int] = None
    failure_width: Optional[int] = None
    failure_reason: Optional[str] = None

    def record_success(width: int, result: ProbeResult) -> None:
        nonlocal best_width, best_peak, best_allocated, best_reserved
        best_width = int(width)
        best_peak = int(result.peak_mib)
        best_allocated = int(result.peak_allocated_mib)
        best_reserved = int(result.peak_reserved_mib)

    result = probe(start_width)
    if result.success:
        record_success(start_width, result)
        low = start_width
        step = width_step
        while True:
            candidate = align_width(low + step, min_width, width_step)
            if candidate <= low or (max_width is not None and candidate > max_width):
                failure_width = None
                failure_reason = None
                break
            result = probe(candidate)
            if result.success:
                record_success(candidate, result)
                low = candidate
                step *= 2
                continue
            failure_width = candidate
            failure_reason = str(result.reason)
            break
    else:
        failure_reason = str(result.reason)
        high = start_width
        step = width_step
        low = None
        while True:
            candidate = align_width(high - step, min_width, width_step)
            if candidate >= high:
                break
            result = probe(candidate)
            if result.success:
                record_success(candidate, result)
                low = candidate
                break
            high = candidate
            step *= 2
            if candidate <= min_width:
                break

        if best_width is None:
            return None, failure_width, failure_reason, best_peak, best_allocated, best_reserved

        if low is None:
            return best_width, failure_width, failure_reason, best_peak, best_allocated, best_reserved

        if failure_width is None:
            failure_width = high
            if failure_reason is None:
                failure_reason = "search_bracket_unknown"

    if best_width is None:
        return None, failure_width, failure_reason, best_peak, best_allocated, best_reserved

    if failure_width is None:
        return best_width, failure_width, failure_reason, best_peak, best_allocated, best_reserved

    lo = int(best_width if result.success else low)
    hi = int(failure_width)
    if hi < lo:
        lo, hi = hi, lo

    while hi - lo > width_step:
        mid = align_width(lo + ((hi - lo) // (2 * width_step)) * width_step, min_width, width_step)
        if mid <= lo:
            mid = align_width(lo + width_step, min_width, width_step)
        if mid >= hi:
            break
        result = probe(mid)
        if result.success:
            record_success(mid, result)
            lo = mid
        else:
            hi = mid
            failure_width = mid
            failure_reason = str(result.reason)

    return best_width, failure_width, failure_reason, best_peak, best_allocated, best_reserved


def log_line(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def batch_size_for_task(task_name: str, default_batch_size: int, overrides: Dict[str, int]) -> int:
    return int(overrides.get(task_name.lower(), int(default_batch_size)))


def run_candidate(
    *,
    task_name: str,
    task: Any,
    depth: int,
    width: int,
    batch_size: int,
    device: torch.device,
    use_bn: bool,
    threshold_mib: int,
    target_epochs: int = 0,
) -> ProbeResult:
    cleanup_cuda()
    model = None
    optimizer = None
    peak_mib = 0
    peak_allocated_mib = 0
    peak_reserved_mib = 0
    batches_completed = 0
    epochs_completed = 0
    reason = "ok"

    try:
        model = MLP(
            in_dim=task.in_dim,
            hidden_widths=[int(width) for _ in range(int(depth))],
            out_dim=task.out_dim,
            use_bn=use_bn,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        if int(target_epochs) > 0:
            for _ in range(int(target_epochs)):
                for batch in task.train_loader:
                    x, y, _ = unpack_batch(batch)
                    x = x.to(device, non_blocking=False)
                    y = y.to(device, non_blocking=False)
                    optimizer.zero_grad(set_to_none=True)
                    out = model(x)
                    loss = task.loss_fn(out, y)
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    batches_completed += 1

                    used_mib = query_gpu_memory_used_mib(torch.cuda.current_device()) or 0
                    allocated_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
                    reserved_mib = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
                    peak_mib = max(peak_mib, used_mib)
                    peak_allocated_mib = max(peak_allocated_mib, allocated_mib)
                    peak_reserved_mib = max(peak_reserved_mib, reserved_mib)
                    if used_mib > threshold_mib:
                        reason = "vram_threshold_exceeded"
                        break
                epochs_completed += 1
                if reason != "ok":
                    break
        else:
            iterator = iter(task.train_loader)
            for _ in range(2):
                try:
                    batch = next(iterator)
                except StopIteration:
                    reason = "train_loader_exhausted"
                    break
                x, y, _ = unpack_batch(batch)
                x = x.to(device, non_blocking=False)
                y = y.to(device, non_blocking=False)
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = task.loss_fn(out, y)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                batches_completed += 1

                used_mib = query_gpu_memory_used_mib(torch.cuda.current_device()) or 0
                allocated_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
                reserved_mib = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
                peak_mib = max(peak_mib, used_mib)
                peak_allocated_mib = max(peak_allocated_mib, allocated_mib)
                peak_reserved_mib = max(peak_reserved_mib, reserved_mib)
                if used_mib > threshold_mib:
                    reason = "vram_threshold_exceeded"
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            reason = "oom"
        else:
            reason = f"runtime_error:{exc.__class__.__name__}"
    finally:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        del task
        cleanup_cuda()

    if int(target_epochs) > 0:
        success = epochs_completed >= int(target_epochs) and peak_mib <= threshold_mib and reason == "ok"
        if epochs_completed < int(target_epochs) and reason == "ok":
            reason = "insufficient_epochs"
    else:
        success = batches_completed >= 2 and peak_mib <= threshold_mib and reason == "ok"
        if batches_completed < 2 and reason == "ok":
            reason = "insufficient_batches"
    if peak_mib > threshold_mib and reason == "ok":
        reason = "vram_threshold_exceeded"
    return ProbeResult(
        task=task_name,
        depth=int(depth),
        width=int(width),
        success=bool(success),
        reason=reason,
        batches_completed=int(batches_completed),
        epochs_completed=int(epochs_completed),
        peak_mib=int(peak_mib),
        peak_allocated_mib=int(peak_allocated_mib),
        peak_reserved_mib=int(peak_reserved_mib),
    )


def main() -> None:
    bootstrap_runtime("probe_width_capacity")

    args = parse_args()
    tasks = [str(t).lower() for t in args.tasks]
    task_batch_sizes = parse_task_batch_sizes(args.task_batch_size)
    run_root = Path(args.run_root) if args.run_root else Path(args.results_dir) / f"width_capacity_probe_{now_stamp()}"
    if args.clear_results and run_root.exists():
        subprocess.run(["rm", "-rf", str(run_root)], check=True)
    run_root.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    summary_rows: List[Dict[str, Any]] = []
    report: Dict[str, Dict[int, Dict[str, Any]]] = {}
    task_cache: Dict[str, Any] = {}

    for task_name in tasks:
        terminate_stale_processes(exclude_pids=[os.getpid()])
        cleanup_cuda()
        log_line(f"task start: {task_name}")
        report[task_name] = {}
        previous_best_width: Optional[int] = None
        task_batch_size = batch_size_for_task(task_name, int(args.batch_size), task_batch_sizes)
        if task_name not in task_cache:
            log_line(f"building task: {task_name}")
            task_cache[task_name] = build_task(task_name, args.data_dir, task_batch_size, int(args.num_workers), int(args.seed))
            refresh_task_loaders(task_cache[task_name], task_batch_size)
        task = task_cache[task_name]
        for depth in range(int(args.min_depth), int(args.max_depth) + 1):
            log_line(f"depth start: task={task_name} depth={depth}")
            calibrated_limit = max(int(args.min_width), int(args.calibrated_limit_mib))
            center_width = max(
                int(args.min_width),
                (calibrated_limit // int(depth) // int(args.width_step)) * int(args.width_step),
            )
            start_width = previous_best_width if previous_best_width is not None else center_width
            log_line(
                f"width center: task={task_name} depth={depth} center={center_width} start={start_width} "
                f"margin={int(args.search_margin)} step={int(args.width_step)}"
            )
            def probe(width: int) -> ProbeResult:
                log_line(
                    f"candidate start: task={task_name} depth={depth} width={width} "
                    f"batch_size={task_batch_size} threshold_mib={int(args.vram_threshold_mib)}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = run_candidate(
                    task_name=task_name,
                    task=task,
                    depth=depth,
                    width=width,
                    batch_size=task_batch_size,
                    device=device,
                    use_bn=bool(args.use_bn),
                    threshold_mib=int(args.vram_threshold_mib),
                    target_epochs=2 if task_name == "simulation" else 0,
                )
                summary_rows.append(asdict(result))
                log_line(
                    f"candidate result: task={task_name} depth={depth} width={width} "
                    f"success={result.success} reason={result.reason} batches={result.batches_completed} "
                    f"peak_mib={result.peak_mib}"
                )
                return result

            best_width, failure_width, failure_reason, best_peak, best_allocated, best_reserved = binary_search_max_width(
        probe=probe,
        start_width=int(start_width),
        min_width=int(args.min_width),
        max_width=None if int(args.max_width) <= 0 else int(args.max_width),
        width_step=int(args.width_step),
    )
            previous_best_width = best_width

            report[task_name][depth] = {
                "max_width": best_width,
                "peak_mib": best_peak,
                "peak_allocated_mib": best_allocated,
                "peak_reserved_mib": best_reserved,
                "failure_width": failure_width,
                "failure_reason": failure_reason,
                "batch_size": int(task_batch_size),
                "vram_threshold_mib": int(args.vram_threshold_mib),
                "epochs_completed": 2 if task_name == "simulation" and best_width is not None else 0,
            }
            log_line(
                f"depth done: task={task_name} depth={depth} max_width={best_width} "
                f"failure_width={failure_width} failure_reason={failure_reason}"
            )

        if torch.cuda.is_available():
            cleanup_cuda()
        log_line(f"task done: {task_name}")

    def summary_markdown() -> str:
        lines: List[str] = []
        lines.append("# Width Capacity Probe Summary")
        lines.append("")
        lines.append(f"- Run root: `{run_root}`")
        lines.append(f"- VRAM threshold: `{int(args.vram_threshold_mib)} MiB`")
        lines.append(f"- Batch size overrides: `{task_batch_sizes}`")
        lines.append("")
        for task_name, depths in report.items():
            lines.append(f"## {task_name}")
            lines.append("")
            lines.append("| depth | max_width | failure_width | failure_reason | peak_mib | batch_size |")
            lines.append("| --- | ---: | ---: | --- | ---: | ---: |")
            for depth, row in sorted(depths.items()):
                lines.append(
                    f"| {depth} | {row['max_width']} | {row['failure_width']} | {row['failure_reason']} | "
                    f"{row['peak_mib']} | {row['batch_size']} |"
                )
            lines.append("")
        return "\n".join(lines)

    (run_root / "probe_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (run_root / "probe_summary.csv").open("w", encoding="utf-8") as f:
        f.write("task,depth,max_width,peak_mib,peak_allocated_mib,peak_reserved_mib,failure_width,failure_reason,batch_size,vram_threshold_mib,epochs_completed\n")
        for task_name, depths in report.items():
            for depth, row in sorted(depths.items()):
                f.write(
                    f"{task_name},{depth},{row['max_width']},{row['peak_mib']},{row['peak_allocated_mib']},{row['peak_reserved_mib']},"
                    f"{row['failure_width']},{row['failure_reason']},{row['batch_size']},{row['vram_threshold_mib']},{row.get('epochs_completed', '')}\n"
                )
    (run_root / "probe_rows.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    (run_root / "probe_summary.md").write_text(summary_markdown(), encoding="utf-8")

    print(json.dumps({"run_root": str(run_root), "summary": report}, indent=2))


if __name__ == "__main__":
    main()
