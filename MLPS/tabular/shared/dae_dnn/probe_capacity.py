from __future__ import annotations

"""Generic GPU capacity probe for arbitrary task/model factories.

This script probes the largest width per depth that satisfies a user-defined
success criterion while staying under a VRAM ceiling. It is intentionally
factory-driven:

- the task factory supplies the dataset/task bundle, loaders, loss, and dims
- the model factory builds the candidate model for a given depth and width

Use it for:
- any dataset the task factory can build
- any model the model factory can instantiate
- any GPU that exposes `nvidia-smi`
- any success horizon, measured in batches or epochs

The probe uses exponential bracketing followed by binary search. Batch size can
be overridden globally or per task, which lets the same script probe different
datasets with different loader shapes.
"""

import argparse
import csv
import gc
import importlib
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from MLPS.tabular.shared.dae_dnn.train_utils import unpack_batch


DEFAULT_TASKS = ["representation", "anomaly", "simulation"]
DEFAULT_VRAM_THRESHOLD_MIB = 6144
DEFAULT_MIN_DEPTH = 1
DEFAULT_MAX_DEPTH = 10
DEFAULT_MIN_WIDTH = 16
DEFAULT_WIDTH_STEP = 16
DEFAULT_SUCCESS_COUNT = 2


@dataclass
class TaskBundle:
    name: str
    train_loader: Any
    val_loader: Any = None
    test_loader: Any = None
    in_dim: int = 0
    out_dim: int = 0
    task_type: str = "regression"
    loss_fn: Optional[Callable] = None
    metrics_fn: Optional[Callable] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateResult:
    task: str
    depth: int
    width: int
    success: bool
    reason: str
    success_unit: str
    success_count: int
    batches_completed: int
    epochs_completed: int
    peak_mib: int
    samples_mib: List[int]
    average_mib: Optional[float]
    model_factory: str
    task_factory: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic GPU capacity probe with exponential bracketing and binary search."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Task names to probe. Passed to the task factory one at a time.",
    )
    parser.add_argument(
        "--task-factory",
        default="MLPS.tabular.shared.dae_dnn.tasks:build_task",
        help="Import path to a callable that builds a task bundle. Format: module:function",
    )
    parser.add_argument(
        "--task-factory-kwargs-json",
        default="{}",
        help="JSON object merged into every task-factory call, e.g. '{\"data_dir\":\"./data\",\"num_workers\":0}'.",
    )
    parser.add_argument(
        "--model-factory",
        default="MLPS.tabular.shared.dae_dnn.mlp:MLP",
        help="Import path to a callable that builds the model. Format: module:function",
    )
    parser.add_argument(
        "--model-factory-kwargs-json",
        default="{}",
        help="JSON object merged into every model-factory call.",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--task-batch-size",
        action="append",
        default=[],
        help="Per-task batch size override in the form task=batch_size. May be repeated.",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Default batch size override. 0 defers to the task builder.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-bn", action="store_true", default=True)
    parser.add_argument("--no-bn", dest="use_bn", action="store_false")
    parser.add_argument("--min-depth", type=int, default=DEFAULT_MIN_DEPTH)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--min-width", type=int, default=DEFAULT_MIN_WIDTH)
    parser.add_argument("--width-step", type=int, default=DEFAULT_WIDTH_STEP)
    parser.add_argument(
        "--width-cut-pct",
        type=float,
        default=10.0,
        help="Reduce every candidate width by this percentage before alignment, then round down to the width step.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Maximum width to test. Use 0 or a negative value for no explicit ceiling.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["exponential_binary"],
        default="exponential_binary",
        help="Search strategy. The current implementation uses exponential bracketing followed by binary search.",
    )
    parser.add_argument(
        "--success-unit",
        choices=["batches", "epochs"],
        default="batches",
        help="Count success by batches or by full epochs.",
    )
    parser.add_argument(
        "--success-count",
        type=int,
        default=DEFAULT_SUCCESS_COUNT,
        help="Number of successful batches or epochs required for a candidate to pass.",
    )
    parser.add_argument(
        "--vram-threshold-mib",
        type=int,
        default=DEFAULT_VRAM_THRESHOLD_MIB,
        help="Fail a candidate if sampled GPU memory exceeds this threshold in MiB.",
    )
    parser.add_argument("--clear-results", action="store_true", default=False)
    parser.add_argument(
        "--kill-stale",
        action="store_true",
        default=True,
        help="Terminate stale probe/training processes before each task starts.",
    )
    parser.add_argument("--no-kill-stale", dest="kill_stale", action="store_false")
    parser.add_argument(
        "--save-samples",
        action="store_true",
        default=True,
        help="Persist per-batch VRAM samples in the JSON report.",
    )
    parser.add_argument("--no-save-samples", dest="save_samples", action="store_false")
    parser.add_argument("--candidate-timeout-sec", type=float, default=0.0, help="Reserved for future use.")
    return parser.parse_args()


def load_callable(spec: str) -> Callable:
    if ":" not in spec:
        raise ValueError(f"Callable spec must be module:function, got: {spec}")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"{spec} is not callable")
    return func


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = str(text).strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    return value


def parse_task_batch_sizes(values: Sequence[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for value in values:
        text = str(value).strip()
        if not text or "=" not in text:
            continue
        task_name, batch_text = text.split("=", 1)
        task_name = task_name.strip().lower()
        try:
            batch_size = max(1, int(batch_text.strip()))
        except Exception:
            continue
        if task_name:
            mapping[task_name] = batch_size
    return mapping


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    gc.collect()


def sample_gpu_mib(device_index: int = 0) -> Optional[int]:
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


def terminate_stale_processes(exclude_pids: Optional[Sequence[int]] = None) -> None:
    exclude = {int(pid) for pid in (exclude_pids or [])}
    patterns = [
        "probe_capacity.py",
        "measure_stl_width_vram.py",
        "probe_width_capacity.py",
        "run_stl_ablation.py",
        "run_stl_ablation_parallel.py",
        "run_with_watchdog.py",
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
        except Exception:
            pass
    if pids_to_kill:
        time.sleep(1.0)
    for pid in sorted(pids_to_kill):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def normalize_task_bundle(task_name: str, raw_task: Any) -> TaskBundle:
    if isinstance(raw_task, TaskBundle):
        return raw_task
    if is_dataclass(raw_task):
        payload = asdict(raw_task)
    elif isinstance(raw_task, dict):
        payload = dict(raw_task)
    elif hasattr(raw_task, "__dict__"):
        payload = dict(vars(raw_task))
    else:
        payload = {key: getattr(raw_task, key) for key in dir(raw_task) if not key.startswith("_")}
    name = str(payload.get("name", task_name))
    return TaskBundle(
        name=name,
        train_loader=payload["train_loader"],
        val_loader=payload.get("val_loader"),
        test_loader=payload.get("test_loader"),
        in_dim=int(payload.get("in_dim", 0)),
        out_dim=int(payload.get("out_dim", 0)),
        task_type=str(payload.get("task_type", "regression")),
        loss_fn=payload.get("loss_fn"),
        metrics_fn=payload.get("metrics_fn"),
        extra=dict(payload.get("extra", {})),
    )


def build_task_bundle(
    task_factory: Callable,
    task_name: str,
    batch_size: int,
    base_kwargs: Dict[str, Any],
) -> TaskBundle:
    kwargs = dict(base_kwargs)
    kwargs.update({"task_name": task_name, "batch_size": int(batch_size)})
    try:
        raw = task_factory(**kwargs)
    except TypeError:
        raw = task_factory(task_name, int(batch_size), **base_kwargs)
    return normalize_task_bundle(task_name, raw)


def build_model(
    model_factory: Callable,
    task: TaskBundle,
    depth: int,
    width: int,
    use_bn: bool,
    model_kwargs: Dict[str, Any],
) -> torch.nn.Module:
    attempts = [
        lambda: model_factory(task=task, depth=int(depth), width=int(width), use_bn=bool(use_bn), **model_kwargs),
        lambda: model_factory(task, int(depth), int(width), use_bn=bool(use_bn), **model_kwargs),
        lambda: model_factory(task.in_dim, [int(width)] * int(depth), task.out_dim, use_bn=bool(use_bn), **model_kwargs),
        lambda: model_factory(task.in_dim, [int(width)] * int(depth), task.out_dim, **model_kwargs),
        lambda: model_factory(in_dim=task.in_dim, hidden_widths=[int(width)] * int(depth), out_dim=task.out_dim, use_bn=bool(use_bn), **model_kwargs),
    ]
    last_exc: Optional[BaseException] = None
    for attempt in attempts:
        try:
            model = attempt()
            if not isinstance(model, torch.nn.Module):
                raise TypeError(f"Model factory returned {type(model)!r}, expected torch.nn.Module")
            return model
        except Exception as exc:
            last_exc = exc
    if last_exc is None:
        raise RuntimeError("Model factory failed without an exception")
    raise last_exc


def align_width(width: int, min_width: int, step: int) -> int:
    width = max(int(min_width), int(width))
    step = max(1, int(step))
    width = int(width // step * step)
    return max(int(min_width), width)


def apply_width_cut(width: int, cut_pct: float, min_width: int, step: int) -> int:
    cut_pct = max(0.0, float(cut_pct))
    if cut_pct <= 0.0:
        return align_width(width, min_width, step)
    reduced = float(width) * (1.0 - cut_pct / 100.0)
    return align_width(int(reduced), min_width, step)


def run_candidate(
    *,
    task_name: str,
    task: TaskBundle,
    task_factory_name: str,
    model_factory: Callable,
    model_factory_name: str,
    model_kwargs: Dict[str, Any],
    depth: int,
    width: int,
    success_unit: str,
    success_count: int,
    vram_threshold_mib: int,
    device: torch.device,
    use_bn: bool,
    save_samples: bool,
) -> CandidateResult:
    cleanup_cuda()
    model = build_model(model_factory, task, depth, width, use_bn, model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    batches_done = 0
    epochs_done = 0
    samples: List[int] = []
    peak_mib = 0
    reason = "ok"

    def record_sample() -> None:
        nonlocal peak_mib
        used_mib = sample_gpu_mib(torch.cuda.current_device() if torch.cuda.is_available() else 0)
        if used_mib is None:
            return
        peak_mib = max(peak_mib, int(used_mib))
        if save_samples:
            samples.append(int(used_mib))
        if int(used_mib) > int(vram_threshold_mib):
            raise RuntimeError("vram_threshold_exceeded")

    try:
        while True:
            batches_this_epoch = 0
            iterator = iter(task.train_loader)
            while True:
                try:
                    batch = next(iterator)
                except StopIteration:
                    break

                x, y, _ = unpack_batch(batch)
                x = x.to(device)
                if y is not None:
                    y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                target = x if task.task_type == "reconstruction" else y
                if target is None:
                    raise ValueError(f"Task {task_name} did not provide a training target")
                loss = task.loss_fn(out, target)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()

                batches_done += 1
                batches_this_epoch += 1
                record_sample()

                if success_unit == "batches" and batches_done >= int(success_count):
                    return CandidateResult(
                        task=task_name,
                        depth=int(depth),
                        width=int(width),
                        success=True,
                        reason="ok",
                        success_unit=success_unit,
                        success_count=int(success_count),
                        batches_completed=int(batches_done),
                        epochs_completed=int(epochs_done),
                        peak_mib=int(peak_mib),
                        samples_mib=list(samples),
                        average_mib=float(sum(samples) / max(len(samples), 1)) if samples else None,
                        model_factory=model_factory_name,
                        task_factory=task_factory_name,
                    )

            if batches_this_epoch == 0:
                reason = "train_loader_exhausted"
                break

            epochs_done += 1
            if success_unit == "epochs" and epochs_done >= int(success_count):
                return CandidateResult(
                    task=task_name,
                    depth=int(depth),
                    width=int(width),
                    success=True,
                    reason="ok",
                    success_unit=success_unit,
                    success_count=int(success_count),
                    batches_completed=int(batches_done),
                    epochs_completed=int(epochs_done),
                    peak_mib=int(peak_mib),
                    samples_mib=list(samples),
                    average_mib=float(sum(samples) / max(len(samples), 1)) if samples else None,
                    model_factory=model_factory_name,
                    task_factory=task_factory_name,
                )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            reason = "oom"
        elif "vram_threshold_exceeded" in msg:
            reason = "vram_threshold_exceeded"
        else:
            reason = f"runtime_error:{exc.__class__.__name__}"
    finally:
        del model
        del optimizer
        cleanup_cuda()

    return CandidateResult(
        task=task_name,
        depth=int(depth),
        width=int(width),
        success=False,
        reason=reason,
        success_unit=success_unit,
        success_count=int(success_count),
        batches_completed=int(batches_done),
        epochs_completed=int(epochs_done),
        peak_mib=int(peak_mib),
        samples_mib=list(samples),
        average_mib=float(sum(samples) / max(len(samples), 1)) if samples else None,
        model_factory=model_factory_name,
        task_factory=task_factory_name,
    )


def binary_search_max_width(
    *,
    probe: Callable[[int], CandidateResult],
    min_width: int,
    width_step: int,
    width_cut_pct: float,
    max_width: Optional[int],
) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[int], Optional[float]]:
    min_width = int(min_width)
    width_step = max(1, int(width_step))
    max_width = None if max_width is None or int(max_width) <= 0 else int(max_width)

    def align(v: int) -> int:
        return apply_width_cut(v, width_cut_pct, min_width, width_step)

    start = align(min_width)
    if max_width is not None:
        start = min(start, max_width)

    start_result = probe(start)
    if start_result.success:
        best_width = start
        best_peak = int(start_result.peak_mib)
        best_avg = start_result.average_mib
        low = start
        step = width_step
        failure_width = None
        failure_reason = None
        while True:
            candidate = align(low + step)
            if candidate <= low or (max_width is not None and candidate > max_width):
                break
            result = probe(candidate)
            if result.success:
                best_width = candidate
                best_peak = int(result.peak_mib)
                best_avg = result.average_mib
                low = candidate
                step *= 2
                continue
            failure_width = candidate
            failure_reason = result.reason
            break

        if failure_width is None:
            return best_width, None, None, best_peak, best_avg

        lo = low
        hi = failure_width
        best_width = lo
        while hi - lo > width_step:
            mid = align((lo + hi) // 2)
            if mid <= lo:
                break
            result = probe(mid)
            if result.success:
                best_width = mid
                best_peak = int(result.peak_mib)
                best_avg = result.average_mib
                lo = mid
            else:
                hi = mid
                failure_width = mid
                failure_reason = result.reason
        return best_width, failure_width, failure_reason, best_peak, best_avg

    failure_reason = start_result.reason
    hi = start
    step = width_step
    lo = None
    best_peak: Optional[int] = None
    best_avg: Optional[float] = None
    while True:
        candidate = align(hi - step)
        if candidate >= hi:
            break
        result = probe(candidate)
        if result.success:
            lo = candidate
            best_peak = int(result.peak_mib)
            best_avg = result.average_mib
            break
        hi = candidate
        step *= 2
        if candidate <= min_width:
            break

    if lo is None:
        return None, hi, failure_reason, best_peak, best_avg

    best_width = lo
    while hi - lo > width_step:
        mid = align((lo + hi) // 2)
        if mid <= lo:
            break
        result = probe(mid)
        if result.success:
            best_width = mid
            best_peak = int(result.peak_mib)
            best_avg = result.average_mib
            lo = mid
        else:
            hi = mid
            failure_reason = result.reason
    return best_width, hi, failure_reason, best_peak, best_avg


def write_reports(run_root: Path, rows: List[Dict[str, Any]]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "probe_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (run_root / "probe_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (run_root / "probe_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})

    md_lines = ["# Capacity Probe Summary", ""]
    current_task = None
    for row in rows:
        if row["task"] != current_task:
            current_task = row["task"]
            md_lines.extend([f"## {current_task}", "", "| depth | max width | failure width | failure reason | peak MiB | avg MiB |", "| --- | ---: | ---: | --- | ---: | ---: |"])
        md_lines.append(
            f"| {row['depth']} | {row.get('max_width')} | {row.get('failure_width')} | {row.get('failure_reason')} | {row.get('peak_mib')} | {row.get('average_mib')} |"
        )
    (run_root / "probe_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    task_factory = load_callable(str(args.task_factory))
    model_factory = load_callable(str(args.model_factory))
    task_factory_kwargs = parse_json_object(args.task_factory_kwargs_json)
    task_factory_kwargs.setdefault("data_dir", args.data_dir)
    task_factory_kwargs.setdefault("num_workers", int(args.num_workers))
    task_factory_kwargs.setdefault("seed", int(args.seed))
    model_factory_kwargs = parse_json_object(args.model_factory_kwargs_json)
    task_batch_sizes = parse_task_batch_sizes(args.task_batch_size)
    run_root = Path(args.run_root) if args.run_root else Path(args.results_dir) / f"capacity_probe_{now_stamp()}"

    if args.clear_results and run_root.exists():
        subprocess.run(["rm", "-rf", str(run_root)], check=True)
    run_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: List[Dict[str, Any]] = []
    start_task_time = time.time()

    for task_name in [str(t).lower() for t in args.tasks]:
        if args.kill_stale:
            terminate_stale_processes(exclude_pids=[os.getpid()])
        cleanup_cuda()

        requested_batch = int(args.batch_size)
        if task_name in task_batch_sizes:
            requested_batch = int(task_batch_sizes[task_name])

        task = build_task_bundle(
            task_factory,
            task_name,
            requested_batch,
            task_factory_kwargs,
        )
        # annotate task factory name for reports
        task_factory_name = str(args.task_factory)

        for depth in range(int(args.min_depth), int(args.max_depth) + 1):
            def probe(width: int) -> CandidateResult:
                return run_candidate(
                    task_name=task_name,
                    task=task,
                    task_factory_name=str(args.task_factory),
                    model_factory=model_factory,
                    model_factory_name=str(args.model_factory),
                    model_kwargs=model_factory_kwargs,
                    depth=depth,
                    width=width,
                    success_unit=str(args.success_unit),
                    success_count=int(args.success_count),
                    vram_threshold_mib=int(args.vram_threshold_mib),
                    device=device,
                    use_bn=bool(args.use_bn),
                    save_samples=bool(args.save_samples),
                )

            max_width, failure_width, failure_reason, peak_mib, avg_mib = binary_search_max_width(
                probe=probe,
                min_width=int(args.min_width),
                width_step=int(args.width_step),
                width_cut_pct=float(args.width_cut_pct),
                max_width=int(args.max_width),
            )

            row = {
                "task": task_name,
                "depth": int(depth),
                "max_width": max_width,
                "failure_width": failure_width,
                "failure_reason": failure_reason,
                "peak_mib": peak_mib,
                "average_mib": avg_mib,
                "success_unit": str(args.success_unit),
                "success_count": int(args.success_count),
                "vram_threshold_mib": int(args.vram_threshold_mib),
                "width_cut_pct": float(args.width_cut_pct),
                "task_batch_size": int(requested_batch),
                "task_factory": task_factory_name,
                "model_factory": str(args.model_factory),
                "elapsed_sec": round(time.time() - start_task_time, 3),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    write_reports(run_root, rows)


if __name__ == "__main__":
    main()
