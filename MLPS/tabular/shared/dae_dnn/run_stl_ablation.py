from __future__ import annotations

import argparse
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

from MLPS.tabular.shared.dae_dnn.tasks import build_task
from utils.adp_logging import ContinuousLogger

import run_goliath as rg


DEFAULT_TASKS = [
    "classification",
    "autoencoding",
    "generation",
    "denoising",
    "anomaly",
    "simulation",
    "prediction",
]

DEFAULT_MIN_DEPTH = 1
DEFAULT_MAX_DEPTH = 10
DEFAULT_MIN_WIDTH = 64
DEFAULT_MAX_WIDTH = 1024
DEFAULT_WIDTH_STEP = 64
DEFAULT_REPEAT_COUNT = 10
DEFAULT_BATCH_TARGETS = {
    "classification": 50,
    "autoencoding": 50,
    "generation": 50,
    "denoising": 50,
    "anomaly": 10,
    "simulation": 1,
    "prediction": 1,
}

CLASSIFICATION_MAX_WIDTH_BY_DEPTH = {
    1: 37376,
    2: 13184,
    3: 10352,
    4: 8432,
    5: 6864,
    6: 5888,
    7: 5168,
    8: 4656,
    9: 4656,
    10: 4336,
}

AUTOENCODING_MAX_WIDTH_BY_DEPTH = {
    1: 29888,
    2: 12544,
    3: 9968,
    4: 8416,
    5: 6864,
    6: 5872,
    7: 5168,
    8: 4608,
    9: 1024,
    10: 256,
}

GENERATION_MAX_WIDTH_BY_DEPTH = {
    1: 37056,
    2: 14272,
    3: 10560,
    4: 8576,
    5: 7296,
    6: 6400,
    7: 5760,
    8: 5248,
    9: 4800,
    10: 4480,
}

DENOISING_MAX_WIDTH_BY_DEPTH = {
    1: 37056,
    2: 14272,
    3: 10560,
    4: 8576,
    5: 7296,
    6: 6400,
    7: 5760,
    8: 5248,
    9: 4800,
    10: 4480,
}

ANOMALY_MAX_WIDTH_BY_DEPTH = {
    1: 20416,
    2: 14272,
    3: 10560,
    4: 8512,
    5: 7296,
    6: 6400,
    7: 5760,
    8: 5248,
    9: 4864,
    10: 4416,
}

SIMULATION_MAX_WIDTH_BY_DEPTH = {
    1: 37568,
    2: 14272,
    3: 10560,
    4: 8576,
    5: 7296,
    6: 6400,
    7: 5760,
    8: 5248,
    9: 4864,
    10: 4480,
}

PREDICTION_MAX_WIDTH_BY_DEPTH = {
    1: 37568,
    2: 14272,
    3: 10560,
    4: 8576,
    5: 7296,
    6: 6400,
    7: 5760,
    8: 5248,
    9: 4864,
    10: 4480,
}

REMAINING_DEPTHS_BY_TASK = {
    "classification": set(range(1, 11)),
    "autoencoding": set(range(1, 11)),
    "generation": set(range(1, 11)),
    "denoising": set(range(1, 11)),
    "anomaly": set(range(1, 11)),
    "simulation": set(range(1, 11)),
    "prediction": set(range(1, 11)),
}


def parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_architectures(values: Sequence[str]) -> List[List[int]]:
    architectures: List[List[int]] = []
    for item in values:
        arch = parse_csv_ints(item)
        if arch:
            architectures.append([max(1, int(v)) for v in arch])
    return architectures


def parse_architecture(text: str) -> List[int]:
    value = str(text).strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [max(1, int(v)) for v in parse_csv_ints(value)]


def dedupe_architectures(architectures: Iterable[Sequence[int]]) -> List[List[int]]:
    seen = set()
    out: List[List[int]] = []
    for arch in architectures:
        key = tuple(int(v) for v in arch)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def build_architectures(args) -> List[List[int]]:
    architecture_arg = getattr(args, "architecture", None)
    widths_arg = getattr(args, "widths", None)
    depths_arg = getattr(args, "depths", None)
    if architecture_arg:
        parsed = dedupe_architectures(parse_architectures(architecture_arg))
        if parsed:
            return parsed
    if widths_arg and depths_arg:
        widths = parse_csv_ints(widths_arg)
        depths = parse_csv_ints(depths_arg)
        return dedupe_architectures([[int(width)] * int(depth) for depth in depths for width in widths])
    min_depth = max(1, int(args.min_depth))
    max_depth = max(min_depth, min(int(args.max_depth), 10))
    if getattr(args, "legacy_architecture_grid", False):
        min_width = max(1, int(args.min_width))
        max_width = max(min_width, int(args.max_width))
        width_step = max(1, int(args.width_step))
        widths = list(range(min_width, max_width + 1, width_step))
        depths = list(range(min_depth, max_depth + 1))
        return dedupe_architectures([[int(width)] * int(depth) for depth in depths for width in widths])
    return [[int(depth)] for depth in range(min_depth, max_depth + 1)]


def phase_name_for_architecture(architecture: Sequence[int], repeat_index: int) -> str:
    depth = len(architecture)
    if depth == 1:
        return f"stl_ablation_r{repeat_index:02d}_d{int(architecture[0]):02d}_parammatched"
    width = max(int(v) for v in architecture)
    return f"stl_ablation_r{repeat_index:02d}_d{depth:02d}_w{width:04d}_{'_'.join(str(int(v)) for v in architecture)}"


def load_source_task_summary(source_run_root: Path, task_name: str) -> Dict[str, Any]:
    path = source_run_root / task_name / "task_summary.json"
    return rg.load_json_if_exists(path) or {}


def comparison_row(
    *,
    task_name: str,
    repeat_index: int,
    ablation_phase: str,
    ablation_architecture: Sequence[int],
    ablation_parameter_count: int,
    ablation_best_val: float,
    ref_phase: str,
    ref_kind: str,
    ref_architecture: Optional[Sequence[int]],
    ref_best_val: float,
) -> Dict[str, Any]:
    winner = "ablation_stl" if float(ablation_best_val) <= float(ref_best_val) else ref_kind
    return {
        "task": task_name,
        "repeat": int(repeat_index),
        "ablation_phase": ablation_phase,
        "ablation_architecture": rg.format_architecture_for_report(ablation_architecture),
        "ablation_parameter_count": int(ablation_parameter_count),
        "ablation_best_val": float(ablation_best_val),
        "reference_kind": ref_kind,
        "reference_phase": ref_phase,
        "reference_architecture": rg.format_architecture_for_report(ref_architecture),
        "reference_best_val": float(ref_best_val),
        "winner": winner,
        "winner_value": min(float(ablation_best_val), float(ref_best_val)),
    }


def parameter_count_for_summary(task: rg.Task, task_root: Path, summary: Dict[str, Any]) -> int:
    phase = str(summary["phase"])
    candidate_dir = task_root / phase / str(summary["candidate_dir"])
    metadata = rg.load_json_if_exists(candidate_dir / "metadata.json") or {}
    model_cfg = metadata.get("model") or {}
    model = rg.make_model(
        int(model_cfg.get("in_dim", task.in_dim)),
        [int(v) for v in model_cfg.get("hidden_widths", summary.get("architecture", []))],
        int(model_cfg.get("out_dim", task.out_dim)),
        bool(model_cfg.get("use_bn", True)),
    )
    return int(rg.count_model_parameters(model))


def target_parameter_count(task: rg.Task, cfg: rg.RunConfig) -> int:
    effective_max_depth = max(1, min(int(cfg.max_depth), 10))
    # Use a quarter of the maximum width as the reference budget so the matched-depth
    # family stays closer to the machine limits while still preserving a single
    # consistent parameter target across depths.
    reference_width = max(1, int(cfg.max_width) // 4)
    reference_architecture = [reference_width for _ in range(effective_max_depth)]
    model = rg.make_stl_model(task, reference_architecture, cfg.use_bn)
    return int(rg.count_model_parameters(model))


def stl_batch_size_for_task(task_name: str, task: rg.Task, override: int, step: int = 16) -> int:
    override = int(override)
    if override > 0:
        return override
    target_batches = max(1, int(DEFAULT_BATCH_TARGETS.get(task_name.lower(), 50)))
    train_rows = len(task.train_loader.dataset)
    min_batch = max(1, int(math.ceil(train_rows / target_batches)))
    rounded = max(int(step), int(math.ceil(min_batch / int(step)) * int(step)))
    return int(rounded)


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


def _parameter_count_for_width(task: rg.Task, depth: int, width: int, cfg: rg.RunConfig) -> int:
    model = rg.make_stl_model(task, [int(width) for _ in range(max(1, int(depth)))], cfg.use_bn)
    return int(rg.count_model_parameters(model))


def solve_parameter_matched_width(task: rg.Task, depth: int, cfg: rg.RunConfig, target_params: int) -> int:
    step = max(1, int(cfg.width_step))
    min_width = max(step, int(cfg.min_width))
    low_units = max(1, int(math.ceil(min_width / step)))
    high_units = max(low_units, int(math.ceil(int(cfg.max_width) / step)))

    best_width = low_units * step
    best_delta = abs(_parameter_count_for_width(task, depth, best_width, cfg) - int(target_params))

    def consider(candidate_width: int) -> None:
        nonlocal best_width, best_delta
        candidate_width = max(step, int(candidate_width))
        delta = abs(_parameter_count_for_width(task, depth, candidate_width, cfg) - int(target_params))
        if delta < best_delta or (delta == best_delta and candidate_width < best_width):
            best_width = candidate_width
            best_delta = delta

    consider(best_width)

    safety_units = max(high_units, low_units) * max(8, max(1, min(int(cfg.max_depth), 10)) * 2)
    while _parameter_count_for_width(task, depth, high_units * step, cfg) < int(target_params) and high_units < safety_units:
        low_units = high_units
        high_units *= 2
        consider(high_units * step)

    while low_units + 1 < high_units:
        mid_units = (low_units + high_units) // 2
        mid_width = mid_units * step
        mid_params = _parameter_count_for_width(task, depth, mid_width, cfg)
        consider(mid_width)
        if mid_params < int(target_params):
            low_units = mid_units
        else:
            high_units = mid_units

    for units in {low_units, high_units, max(1, low_units - 1), high_units + 1}:
        consider(units * step)

    return int(best_width)


def parameter_matched_architecture(task: rg.Task, depth: int, cfg: rg.RunConfig) -> List[int]:
    width = solve_parameter_matched_width(task, depth, cfg, target_parameter_count(task, cfg))
    return [int(width) for _ in range(max(1, int(depth)))]


def parameter_matched_architectures(task: rg.Task, depth: int, cfg: rg.RunConfig) -> List[List[int]]:
    depth = max(1, int(depth))
    step = max(1, int(cfg.width_step))
    min_width = max(step, int(cfg.min_width))
    start_width = int(math.ceil(min_width / step) * step)
    task_name = str(getattr(task, "name", "")).lower()
    if depth not in REMAINING_DEPTHS_BY_TASK.get(task_name, set()):
        return []
    if task_name == "classification":
        max_width = int(
            CLASSIFICATION_MAX_WIDTH_BY_DEPTH.get(depth, CLASSIFICATION_MAX_WIDTH_BY_DEPTH[max(CLASSIFICATION_MAX_WIDTH_BY_DEPTH)])
        )
    elif task_name == "autoencoding":
        max_width = int(
            AUTOENCODING_MAX_WIDTH_BY_DEPTH.get(
                depth,
                AUTOENCODING_MAX_WIDTH_BY_DEPTH[max(AUTOENCODING_MAX_WIDTH_BY_DEPTH)],
            )
        )
    elif task_name == "generation":
        max_width = int(
            GENERATION_MAX_WIDTH_BY_DEPTH.get(
                depth,
                GENERATION_MAX_WIDTH_BY_DEPTH[max(GENERATION_MAX_WIDTH_BY_DEPTH)],
            )
        )
    elif task_name == "denoising":
        max_width = int(
            DENOISING_MAX_WIDTH_BY_DEPTH.get(
                depth,
                DENOISING_MAX_WIDTH_BY_DEPTH[max(DENOISING_MAX_WIDTH_BY_DEPTH)],
            )
        )
    elif task_name == "anomaly" and depth in ANOMALY_MAX_WIDTH_BY_DEPTH:
        max_width = int(ANOMALY_MAX_WIDTH_BY_DEPTH[depth])
    elif task_name == "simulation":
        max_width = int(SIMULATION_MAX_WIDTH_BY_DEPTH.get(depth, SIMULATION_MAX_WIDTH_BY_DEPTH[max(SIMULATION_MAX_WIDTH_BY_DEPTH)]))
    elif task_name == "prediction":
        max_width = int(PREDICTION_MAX_WIDTH_BY_DEPTH.get(depth, PREDICTION_MAX_WIDTH_BY_DEPTH[max(PREDICTION_MAX_WIDTH_BY_DEPTH)]))
    else:
        max_width = solve_parameter_matched_width(task, depth, cfg, target_parameter_count(task, cfg))
    max_width = int(math.floor(max_width / step) * step)
    if max_width < start_width:
        max_width = start_width
    widths = list(range(max_width, start_width - 1, -step))
    return [[int(width) for _ in range(depth)] for width in widths]


def make_cfg(args, tasks: List[str], run_root: Path) -> rg.RunConfig:
    return rg.RunConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        run_root=str(run_root),
        tasks=tasks,
        phases=[],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        stl_width=int(args.stl_width),
        stl_depth=int(args.stl_depth),
        alt_start_width=1,
        alt_start_depth=1,
        patience=int(args.patience),
        width_expansion_patience=10,
        depth_expansion_patience=2,
        delta=float(args.delta),
        max_epochs=int(args.max_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        max_width=int(args.max_width),
        max_depth=int(args.max_depth),
        max_neurons=int(args.max_neurons),
        width_stage_margin_patience=int(args.width_stage_margin_patience),
        width_stage_min_improve_pct=float(args.width_stage_min_improve_pct),
        use_bn=bool(args.use_bn),
        demo=False,
        metrics_every=int(args.metrics_every),
        min_width=int(args.min_width),
        width_step=int(args.width_step),
        parameter_matched=not bool(getattr(args, "legacy_architecture_grid", False)),
    )


def ablation_state_path(task_root: Path) -> Path:
    return task_root / "ablation_state.json"


def load_ablation_state(task_root: Path) -> Dict[str, Any]:
    data = rg.load_json_if_exists(ablation_state_path(task_root))
    return data if isinstance(data, dict) else {}


def save_ablation_state(task_root: Path, state: Dict[str, Any]) -> None:
    rg.write_json(ablation_state_path(task_root), state)


def current_candidate_failed(task_root: Path, saved_state: Dict[str, Any]) -> bool:
    phase_name = str(saved_state.get("current_phase_name") or "").strip()
    if not phase_name:
        return False
    candidate_state_paths = sorted(task_root.rglob(f"{phase_name}/cand_*/candidate_state.json"))
    for candidate_state_path in candidate_state_paths:
        state = rg.load_json_if_exists(candidate_state_path) or {}
        if bool(state.get("failed", False)):
            return True
    return False


def run_task_ablation(
    *,
    task_name: str,
    task_root: Path,
    cfg: rg.RunConfig,
    pin_memory: bool,
    source_run_root: Path,
    architectures: Sequence[Sequence[int]],
    repeat_count: int,
    repeat_index: Optional[int],
    device,
    log: ContinuousLogger,
    batch_controller,
) -> Dict[str, Any]:
    task = build_task(
        task_name,
        cfg.data_dir,
        1,
        cfg.num_workers,
        cfg.seed,
        pin_memory=bool(pin_memory),
    )
    task_batch_size = stl_batch_size_for_task(task_name, task, cfg.batch_size)
    rg.refresh_task_loaders(task, task_batch_size)
    source_summary = load_source_task_summary(source_run_root, task_name)
    target_params = target_parameter_count(task, cfg) if cfg.parameter_matched else None

    ablation_runs: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    source_adp_runs = list(source_summary.get("adp_runs", []))
    source_paired_stl_runs = list(source_summary.get("paired_stl_runs", []))
    source_refs = [
        ("adp", entry.get("phase"), entry.get("architecture"), float(entry.get("best_val", float("inf"))))
        for entry in source_adp_runs
    ] + [
        ("paired_stl", entry.get("phase"), entry.get("architecture"), float(entry.get("best_val", float("inf"))))
        for entry in source_paired_stl_runs
    ]

    best_ablation: Optional[Dict[str, Any]] = None
    repeat_count = max(1, int(repeat_count))
    curve_rows: List[Dict[str, Any]] = []
    repeat_indices = [int(repeat_index)] if repeat_index is not None else list(range(1, repeat_count + 1))
    gpu_vram_samples_mib: List[int] = []

    sampler_stop = threading.Event()

    def sample_gpu_vram() -> None:
        if not torch.cuda.is_available():
            return
        device_index = int(torch.cuda.current_device())
        for _ in range(5):
            if sampler_stop.is_set():
                break
            sample = query_gpu_memory_used_mib(device_index)
            if sample is not None:
                gpu_vram_samples_mib.append(int(sample))
            if len(gpu_vram_samples_mib) >= 5:
                break
            time.sleep(1.0)

    sampler_thread = threading.Thread(target=sample_gpu_vram, daemon=True)
    sampler_thread.start()

    saved_state = load_ablation_state(task_root)
    resume_arch_idx = max(0, int(saved_state.get("architecture_index", 0) or 0))
    resume_family_idx = max(0, int(saved_state.get("family_index", 0) or 0))
    resume_repeat_idx = max(1, int(saved_state.get("repeat_index", 1) or 1))
    resume_completed = bool(saved_state.get("completed", False))
    if resume_completed and (task_root / "ablation_summary.json").exists():
        return rg.load_json_if_exists(task_root / "ablation_summary.json") or {}
    if current_candidate_failed(task_root, saved_state):
        resume_family_idx = int(resume_family_idx) + 1
        resume_repeat_idx = 1
        saved_state.update(
            {
                "architecture_index": int(resume_arch_idx),
                "family_index": int(resume_family_idx),
                "repeat_index": int(resume_repeat_idx),
                "repeat_count": int(repeat_count),
                "completed": False,
                "skipped_failed_family": True,
            }
        )
        save_ablation_state(task_root, saved_state)

    for architecture_idx, architecture in enumerate(architectures):
        family = [list(architecture)]
        if cfg.parameter_matched and len(architecture) == 1:
            family = parameter_matched_architectures(task, int(architecture[0]), cfg)
        if not family:
            continue
        family_start_idx = resume_family_idx if architecture_idx == resume_arch_idx else 0
        for family_idx, expanded_architecture in enumerate(family):
            if architecture_idx == resume_arch_idx and family_idx < family_start_idx:
                continue
            repeat_start = repeat_indices[0]
            if architecture_idx == resume_arch_idx and family_idx == resume_family_idx:
                repeat_start = resume_repeat_idx
            for repeat_id in repeat_indices:
                if repeat_index is None and repeat_id < repeat_start:
                    continue
                if repeat_index is None:
                    next_repeat = repeat_id + 1
                    next_family = family_idx
                    next_architecture = architecture_idx
                    if next_repeat > repeat_count:
                        next_repeat = 1
                        next_family = family_idx + 1
                        if next_family >= len(family):
                            next_family = 0
                            next_architecture = architecture_idx + 1
                    save_ablation_state(
                        task_root,
                        {
                            "task": task_name,
                            "architecture_index": int(next_architecture),
                            "family_index": int(next_family),
                            "repeat_index": int(next_repeat),
                            "repeat_count": int(repeat_count),
                            "completed": False,
                            "current_architecture_index": int(architecture_idx),
                            "current_family_index": int(family_idx),
                            "current_repeat_index": int(repeat_id),
                            "current_phase_name": phase_name_for_architecture(expanded_architecture, repeat_id),
                            "current_architecture": [int(v) for v in expanded_architecture],
                        },
                    )
                phase_name = phase_name_for_architecture(expanded_architecture, repeat_id)
                log.log_console(
                    f"[ABLATION:{task_name}] STL phase start: {phase_name} architecture={rg.format_architecture_for_report(expanded_architecture)}"
                )
                summary = rg.run_stl_phase(
                    task,
                    task_root,
                    cfg,
                    device,
                    list(expanded_architecture),
                    phase_name=phase_name,
                    source_phase=None,
                    batch_controller=batch_controller,
                )
                ablation_runs.append(summary)
                parameter_count = parameter_count_for_summary(task, task_root, summary)

                curve_rows.append(
                    {
                        "task": task_name,
                        "repeat": int(repeat_id),
                        "phase": phase_name,
                        "architecture": rg.format_architecture_for_report(expanded_architecture),
                        "parameter_count": int(parameter_count),
                        "best_val": float(summary.get("best_val", float("inf"))),
                        "best_epoch": int(summary.get("best_epoch", 0)),
                        "final_epoch": int(summary.get("final_epoch", summary.get("best_epoch", 0))),
                        "test_loss": (summary.get("test_metrics") or {}).get("test_loss"),
                        "test_acc": (summary.get("test_metrics") or {}).get("test_acc"),
                    }
                )

                if best_ablation is None or float(summary.get("best_val", float("inf"))) < float(best_ablation.get("best_val", float("inf"))):
                    best_ablation = {
                        **summary,
                        "repeat": int(repeat_id),
                        "parameter_count": int(parameter_count),
                    }

                rows.append(
                    {
                        "task": task_name,
                        "repeat": int(repeat_id),
                        "row_type": "ablation_stl",
                        "phase": phase_name,
                        "architecture": rg.format_architecture_for_report(expanded_architecture),
                        "parameter_count": int(parameter_count),
                        "best_val": float(summary.get("best_val", float("inf"))),
                        "best_epoch": int(summary.get("best_epoch", 0)),
                        "final_epoch": int(summary.get("final_epoch", summary.get("best_epoch", 0))),
                        "test_loss": (summary.get("test_metrics") or {}).get("test_loss"),
                        "test_acc": (summary.get("test_metrics") or {}).get("test_acc"),
                    }
                )

                for ref_kind, ref_phase, ref_arch, ref_best_val in source_refs:
                    comparisons.append(
                        comparison_row(
                            task_name=task_name,
                            repeat_index=repeat_id,
                            ablation_phase=phase_name,
                            ablation_architecture=expanded_architecture,
                            ablation_parameter_count=parameter_count,
                            ablation_best_val=float(summary.get("best_val", float("inf"))),
                            ref_phase=str(ref_phase),
                            ref_kind=ref_kind,
                            ref_architecture=ref_arch,
                            ref_best_val=float(ref_best_val),
                        )
                    )

        resume_family_idx = 0
        resume_repeat_idx = 1

    save_ablation_state(
        task_root,
        {
            "task": task_name,
            "architecture_index": len(architectures),
            "family_index": 0,
            "repeat_index": 1,
            "repeat_count": int(repeat_count),
            "completed": True,
        },
    )

    sampler_stop.set()
    sampler_thread.join(timeout=2.0)
    gpu_vram_avg_mib = float(sum(gpu_vram_samples_mib) / max(len(gpu_vram_samples_mib), 1)) if gpu_vram_samples_mib else None
    log.log_console(
        f"[ABLATION:{task_name}] GPU VRAM samples={gpu_vram_samples_mib} avg_mib={gpu_vram_avg_mib}"
    )

    rg.write_csv(
        task_root / "ablation_summary.csv",
        rows,
        fieldnames=["task", "repeat", "row_type", "phase", "architecture", "parameter_count", "best_val", "best_epoch", "final_epoch", "test_loss", "test_acc"],
    )
    rg.write_json(
        task_root / "ablation_summary.json",
        {
            "task": task_name,
            "source_task_summary": str(source_run_root / task_name / "task_summary.json"),
            "source_adp_runs": source_adp_runs,
            "source_paired_stl_runs": source_paired_stl_runs,
            "parameter_matched": bool(cfg.parameter_matched),
            "parameter_budget_target": int(target_params) if target_params is not None else None,
            "gpu_vram_samples_mib": gpu_vram_samples_mib,
            "gpu_vram_avg_mib": gpu_vram_avg_mib,
            "ablation_stl_runs": ablation_runs,
            "comparisons": comparisons,
            "best_ablation": best_ablation,
            "repeat_count": repeat_count,
            "architecture_count": len(architectures),
        },
    )

    if curve_rows:
        plot_task_ablation(task_root, task_name, curve_rows)

    return {
        "task": task_name,
        "source_adp_runs": source_adp_runs,
        "source_paired_stl_runs": source_paired_stl_runs,
        "ablation_stl_runs": ablation_runs,
        "comparisons": comparisons,
        "best_ablation": best_ablation,
        "repeat_count": repeat_count,
        "curve_rows": curve_rows,
        "gpu_vram_samples_mib": gpu_vram_samples_mib,
        "gpu_vram_avg_mib": gpu_vram_avg_mib,
    }


def plot_task_ablation(task_root: Path, task_name: str, rows: Sequence[Dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(18, 12))
    depths = sorted({len(parse_architecture(str(row["architecture"]))) for row in rows})
    cmap = plt.get_cmap("tab10")
    for idx, depth in enumerate(depths):
        depth_rows = [row for row in rows if len(parse_architecture(str(row["architecture"]))) == depth]
        if not depth_rows:
            continue

        grouped: Dict[float, float] = {}
        for row in depth_rows:
            param_count = float(row["parameter_count"])
            best_val = float(row["best_val"])
            prev = grouped.get(param_count)
            if prev is None or best_val < prev:
                grouped[param_count] = best_val

        xs = sorted(grouped.keys())
        ys = [grouped[x] for x in xs]
        color = cmap(idx % 10)
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.9, label=f"depth {depth}")
        ax.scatter(xs, ys, color=color, s=14, alpha=0.65)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("Best validation loss (log scale)")
    ax.set_title(f"{task_name}: STL ablation loss vs parameters", fontsize=16, pad=16)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    plot_path = task_root / "ablation_loss_vs_params.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    rg.write_json(
        task_root / "ablation_plot.json",
        {
            "task": task_name,
            "plot_path": str(plot_path),
            "depths": depths,
            "note": "Each colored line is one hidden-depth family, parameter-matched to the same task-specific budget and aggregated across repeats.",
        },
    )
    return plot_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full STL architecture ablation for selected tabular DAE/DNN tasks.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default=None)
    p.add_argument("--source-run-root", default="MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-width", type=int, default=1024)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--width-stage-margin-patience", type=int, default=10)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--min-width", type=int, default=DEFAULT_MIN_WIDTH)
    p.add_argument("--width-step", type=int, default=DEFAULT_WIDTH_STEP)
    p.add_argument("--min-depth", type=int, default=DEFAULT_MIN_DEPTH)
    p.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
    p.add_argument("--repeat-index", type=int, default=None, help="Run exactly one repeat index, useful for parallel fan-out.")
    p.add_argument("--stl-width", type=int, default=128)
    p.add_argument("--stl-depth", type=int, default=2)
    p.add_argument("--metrics-every", type=int, default=0, help="Run auxiliary task metrics every N epochs; 0 disables them.")
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=False)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.add_argument("--use-bn", action="store_true", default=True)
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    p.add_argument("--widths", default="")
    p.add_argument("--depths", default="")
    p.add_argument("--architecture", action="append", default=[], help="Explicit hidden widths, e.g. --architecture 64,64,64")
    p.add_argument(
        "--legacy-architecture-grid",
        action="store_true",
        default=False,
        help="Use the old fixed width x depth sweep instead of parameter-matched depth families.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [str(t).lower() for t in args.tasks]
    architectures = build_architectures(args)
    if not architectures:
        raise SystemExit("No architectures requested.")

    source_run_root = Path(args.source_run_root)
    run_root = Path(args.run_root) if args.run_root else Path(args.results_dir) / f"stl_ablation_{rg.now_stamp()}"
    run_root.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg(args, tasks, run_root)
    rg.seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_controller = None

    logger = ContinuousLogger(run_root, "stl_ablation", "stl_ablation")
    logger.log_console(f"Run root: {run_root}")
    logger.log_console(f"Tasks: {tasks}")
    logger.log_console(f"Architectures: {[rg.format_architecture_for_report(a) for a in architectures]}")
    logger.log_console(f"Repeat count: {int(args.repeat_count)}")
    logger.log_console(f"Source run root: {source_run_root}")
    logger.log_console(f"Device: {device}")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    task_reports: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    try:
        for task_name in tasks:
            task_root = run_root / task_name
            task_root.mkdir(parents=True, exist_ok=True)
            report = run_task_ablation(
                task_name=task_name,
                task_root=task_root,
                cfg=cfg,
                pin_memory=bool(args.pin_memory),
                source_run_root=source_run_root,
                architectures=architectures,
                repeat_count=int(args.repeat_count),
                repeat_index=args.repeat_index,
                device=device,
                log=logger,
                batch_controller=batch_controller,
            )
            task_reports.append(report)
            comparison_rows.extend(report["comparisons"])
            rg.cleanup_runtime()

        if comparison_rows:
            rg.write_csv(
                run_root / "comparison_summary.csv",
                comparison_rows,
                fieldnames=[
                    "task",
                    "repeat",
                    "ablation_phase",
                    "ablation_architecture",
                    "ablation_parameter_count",
                    "ablation_best_val",
                    "reference_kind",
                    "reference_phase",
                    "reference_architecture",
                    "reference_best_val",
                    "winner",
                    "winner_value",
                ],
            )

        rg.write_json(
            run_root / "comparison_summary.json",
            {
                "tasks": tasks,
                "architectures": architectures,
                "source_run_root": str(source_run_root),
                "repeat_count": int(args.repeat_count),
                "reports": task_reports,
            },
        )
    finally:
        rg.cleanup_runtime()
        logger.close()


if __name__ == "__main__":
    main()
