from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from MLPS.tabular.shared.dae_dnn.mlp import MLP
from MLPS.tabular.shared.dae_dnn.tasks import build_task, refresh_task_loaders
from MLPS.tabular.shared.dae_dnn.train_utils import unpack_batch


DEFAULT_TASKS = [
    "representation",
    "autoencoding",
    "generation",
    "denoising",
    "anomaly",
    "simulation",
    "prediction",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe the largest width per depth that stays under a VRAM threshold for two training batches.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default=None)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--batch-size", type=int, default=40960)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-depth", type=int, default=1)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--min-width", type=int, default=16)
    p.add_argument("--width-step", type=int, default=16)
    p.add_argument("--max-width", type=int, default=16384)
    p.add_argument("--calibrated-limit-mib", type=int, default=9472)
    p.add_argument("--search-margin", type=int, default=128)
    p.add_argument("--vram-threshold-mib", type=int, default=6144)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-bn", action="store_true", default=True)
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    p.add_argument("--clear-results", action="store_true", default=True)
    return p.parse_args()


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
    peak_mib: int
    peak_allocated_mib: int
    peak_reserved_mib: int


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    gc.collect()


def log_line(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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
) -> ProbeResult:
    cleanup_cuda()
    model = MLP(in_dim=task.in_dim, hidden_widths=[int(width) for _ in range(int(depth))], out_dim=task.out_dim, use_bn=use_bn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    peak_mib = 0
    peak_allocated_mib = 0
    peak_reserved_mib = 0
    batches_completed = 0
    reason = "ok"

    try:
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
        del model
        del optimizer
        del task
        cleanup_cuda()

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
        peak_mib=int(peak_mib),
        peak_allocated_mib=int(peak_allocated_mib),
        peak_reserved_mib=int(peak_reserved_mib),
    )


def main() -> None:
    args = parse_args()
    tasks = [str(t).lower() for t in args.tasks]
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
        log_line(f"task start: {task_name}")
        report[task_name] = {}
        if task_name not in task_cache:
            log_line(f"building task: {task_name}")
            task_cache[task_name] = build_task(task_name, args.data_dir, int(args.batch_size), int(args.num_workers), int(args.seed))
            refresh_task_loaders(task_cache[task_name], int(args.batch_size))
        task = task_cache[task_name]
        for depth in range(int(args.min_depth), int(args.max_depth) + 1):
            log_line(f"depth start: task={task_name} depth={depth}")
            best_width: Optional[int] = None
            best_peak: Optional[int] = None
            best_allocated: Optional[int] = None
            best_reserved: Optional[int] = None
            failure_width: Optional[int] = None
            failure_reason: Optional[str] = None
            calibrated_limit = max(int(args.min_width), int(args.calibrated_limit_mib))
            center_width = max(
                int(args.min_width),
                (calibrated_limit // int(depth) // int(args.width_step)) * int(args.width_step),
            )
            log_line(
                f"width center: task={task_name} depth={depth} center={center_width} "
                f"margin={int(args.search_margin)} step={int(args.width_step)}"
            )
            def probe(width: int) -> ProbeResult:
                log_line(
                    f"candidate start: task={task_name} depth={depth} width={width} "
                    f"batch_size={int(args.batch_size)} threshold_mib={int(args.vram_threshold_mib)}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = run_candidate(
                    task_name=task_name,
                    task=task,
                    depth=depth,
                    width=width,
                    batch_size=int(args.batch_size),
                    device=device,
                    use_bn=bool(args.use_bn),
                    threshold_mib=int(args.vram_threshold_mib),
                )
                summary_rows.append(asdict(result))
                log_line(
                    f"candidate result: task={task_name} depth={depth} width={width} "
                    f"success={result.success} reason={result.reason} batches={result.batches_completed} "
                    f"peak_mib={result.peak_mib}"
                )
                return result

            width = int(center_width)
            result = probe(width)
            if result.success:
                best_width = width
                best_peak = int(result.peak_mib)
                best_allocated = int(result.peak_allocated_mib)
                best_reserved = int(result.peak_reserved_mib)
                width += int(args.width_step)
                while width <= int(args.max_width):
                    result = probe(width)
                    if result.success:
                        best_width = width
                        best_peak = int(result.peak_mib)
                        best_allocated = int(result.peak_allocated_mib)
                        best_reserved = int(result.peak_reserved_mib)
                        width += int(args.width_step)
                        continue
                    failure_width = width
                    failure_reason = str(result.reason)
                    break
            else:
                failure_reason = str(result.reason)
                width -= int(args.width_step)
                while width >= int(args.min_width):
                    result = probe(width)
                    if result.success:
                        best_width = width
                        best_peak = int(result.peak_mib)
                        best_allocated = int(result.peak_allocated_mib)
                        best_reserved = int(result.peak_reserved_mib)
                        width += int(args.width_step)
                        break
                    width -= int(args.width_step)
                if best_width is not None and failure_width is None:
                    while width <= int(args.max_width):
                        result = probe(width)
                        if result.success:
                            best_width = width
                            best_peak = int(result.peak_mib)
                            best_allocated = int(result.peak_allocated_mib)
                            best_reserved = int(result.peak_reserved_mib)
                            width += int(args.width_step)
                            continue
                        failure_width = width
                        failure_reason = str(result.reason)
                        break

            report[task_name][depth] = {
                "max_width": best_width,
                "peak_mib": best_peak,
                "peak_allocated_mib": best_allocated,
                "peak_reserved_mib": best_reserved,
                "failure_width": failure_width,
                "failure_reason": failure_reason,
                "batch_size": int(args.batch_size),
                "vram_threshold_mib": int(args.vram_threshold_mib),
            }
            log_line(
                f"depth done: task={task_name} depth={depth} max_width={best_width} "
                f"failure_width={failure_width} failure_reason={failure_reason}"
            )

        if torch.cuda.is_available():
            cleanup_cuda()
        log_line(f"task done: {task_name}")

    (run_root / "probe_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (run_root / "probe_summary.csv").open("w", encoding="utf-8") as f:
        f.write("task,depth,max_width,peak_mib,peak_allocated_mib,peak_reserved_mib,failure_width,failure_reason,batch_size,vram_threshold_mib\n")
        for task_name, depths in report.items():
            for depth, row in sorted(depths.items()):
                f.write(
                    f"{task_name},{depth},{row['max_width']},{row['peak_mib']},{row['peak_allocated_mib']},{row['peak_reserved_mib']},"
                    f"{row['failure_width']},{row['failure_reason']},{row['batch_size']},{row['vram_threshold_mib']}\n"
                )
    (run_root / "probe_rows.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(json.dumps({"run_root": str(run_root), "summary": report}, indent=2))


if __name__ == "__main__":
    main()
