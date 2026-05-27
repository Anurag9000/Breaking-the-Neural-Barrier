from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

sys.path.append(str(Path(__file__).resolve().parents[2]))

from DAE.DNN.adp_search import ADPConfig, adp_search, expand_depth, expand_width
from DAE.DNN.mlp import MLP
from DAE.DNN.run_goliath import (
    RunConfig,
    build_run_root,
    eval_final,
    extract_hidden_widths,
    format_architecture_for_report,
    git_commit,
    load_candidate_model,
    make_reconstruction_model,
    make_stl_model,
    model_signature,
    now_stamp,
    phase_metadata,
    render_final_report as _render_goliath_report,
    training_loop,
    write_csv,
    write_json,
    write_text,
)
from DAE.DNN.tasks import Task, build_task, refresh_task_loaders, task_names
from DAE.DNN.train_utils import AdaptiveBatchController
from utils.adp_logging import ContinuousLogger


DEFAULT_MAX_EPOCHS = 10**18
DEFAULT_VRAM_BUDGET_GB = 5.5
BASE_WIDTH_CANDIDATES = [2, 4, 8, 16, 32, 64, 128, 256, 512]
BASE_DEPTH_CANDIDATES = [2, 3, 4, 5]
ADP_METHODS = [
    "adp_alt_width",
    "adp_alt_depth",
    "adp_width_to_depth",
    "adp_depth_to_width",
]
DEFAULT_BASELINE_METHODS = ["grid", "random", "bayes", "nas"]


@dataclass
class SuiteConfig:
    data_dir: str
    results_dir: str
    run_root: Optional[str]
    tasks: List[str]
    methods: List[str]
    batch_size: int
    num_workers: int
    seed: int
    patience: int
    delta: float
    max_epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    max_width: int
    max_depth: int
    max_neurons: int
    width_stage_margin_patience: int
    width_stage_min_improve_pct: float
    use_bn: bool
    candidate_budget: int
    bayes_warmup: int
    nas_patience: int
    reference_run_root: Optional[str]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_reconstruct(task: Task) -> bool:
    return task.task_type == "reconstruction"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_adp_method(method: str) -> bool:
    return method.startswith("adp_")


def adp_mode_for_method(method: str) -> str:
    mapping = {
        "adp_alt_width": "alt_width",
        "adp_alt_depth": "alt_depth",
        "adp_width_to_depth": "width_to_depth",
        "adp_depth_to_width": "depth_to_width",
    }
    if method not in mapping:
        raise ValueError(f"Unknown ADP method: {method}")
    return mapping[method]


def task_root_for(run_root: Path, task_name: str) -> Path:
    return run_root / task_name


def method_root_for(task_root: Path, method: str) -> Path:
    return task_root / method


def candidate_root_for(method_root: Path, index: int, hidden_widths: Sequence[int]) -> Path:
    depth = len(hidden_widths)
    width = max(hidden_widths) if hidden_widths else 0
    return method_root / f"cand_{index:03d}_d{depth}_w{width}"


def candidate_summary_path(candidate_dir: Path) -> Path:
    return candidate_dir / "candidate_summary.json"


def candidate_state_path(candidate_dir: Path) -> Path:
    return candidate_dir / "candidate_state.json"


def method_summary_path(task_root: Path, method: str) -> Path:
    return method_root_for(task_root, method) / "method_summary.json"


def method_state_path(task_root: Path, method: str) -> Path:
    return method_root_for(task_root, method) / "method_state.json"


def task_state_path(task_root: Path) -> Path:
    return task_root / "task_state.json"


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def load_completed_candidate_summary(candidate_dir: Path) -> Optional[Dict[str, Any]]:
    state = load_json_if_exists(candidate_state_path(candidate_dir))
    summary = load_json_if_exists(candidate_summary_path(candidate_dir))
    if state is not None and bool(state.get("completed", False)) and summary is not None:
        return summary
    return None


def collect_completed_candidate_summaries(method_root: Path, task: Task, device, reconstruct: bool) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for candidate_dir in sorted([p for p in method_root.iterdir() if p.is_dir() and p.name.startswith("cand_")]):
        summary = load_completed_candidate_summary(candidate_dir)
        if summary is not None:
            summaries.append(summary)
            continue
        state = load_json_if_exists(candidate_state_path(candidate_dir))
        if state is None or not bool(state.get("completed", False)):
            continue
        try:
            model, _, ckpt = load_candidate_model(candidate_dir, device)
        except Exception:
            continue
        test_metrics = eval_final(model, task, device, reconstruct=reconstruct)
        hidden_widths = extract_hidden_widths(state.get("architecture") or [])
        if not hidden_widths:
            hidden_widths = list(model.hidden_widths)
        reconstructed = {
            "method": candidate_dir.parent.name,
            "candidate_dir": str(candidate_dir),
            "architecture": {"hidden_widths": hidden_widths, "in_dim": task.in_dim, "out_dim": task.out_dim, "use_bn": getattr(model, "use_bn", True)},
            "best_val": float(ckpt["best_val"]),
            "best_epoch": int(ckpt["best_epoch"]),
            "final_epoch": int(ckpt.get("epoch", ckpt["best_epoch"])),
            "best_checkpoint": str(candidate_dir / "checkpoint_best.pt"),
            "last_checkpoint": str(candidate_dir / "checkpoint_last.pt"),
            "test_metrics": test_metrics,
            "reconstruct": reconstruct,
        }
        write_json(candidate_summary_path(candidate_dir), reconstructed)
        summaries.append(reconstructed)
    return summaries


def load_method_state(task_root: Path, method: str) -> Optional[Dict[str, Any]]:
    return load_json_if_exists(method_state_path(task_root, method))


def save_method_state(task_root: Path, method: str, state: Dict[str, Any]) -> None:
    write_json(method_state_path(task_root, method), state)


def load_task_state(task_root: Path) -> Optional[Dict[str, Any]]:
    return load_json_if_exists(task_state_path(task_root))


def save_task_state(task_root: Path, state: Dict[str, Any]) -> None:
    write_json(task_state_path(task_root), state)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_reference_task_summary(reference_run_root: Optional[str], task_name: str) -> Optional[Dict[str, Any]]:
    if not reference_run_root:
        return None
    task_summary_path = Path(reference_run_root) / task_name / "task_summary.json"
    if not task_summary_path.exists():
        return None
    try:
        return read_json(task_summary_path)
    except Exception:
        return None


def reference_winner_comparison(reference_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    winner = reference_summary.get("winner") or {}
    if not winner:
        return None
    winner_phase = winner.get("winner_phase")
    winner_kind = winner.get("winner")
    winner_value = winner.get("winner_value")
    if winner_phase is None or winner_value is None:
        return None

    winner_architecture = None
    winner_metrics: Dict[str, Any] = {}
    for comp in reference_summary.get("comparisons", []):
        method = comp.get("method")
        if method == winner_phase:
            winner_architecture = comp.get("search_architecture")
            winner_metrics = comp.get("search_test_metrics", {})
            break
        if winner_phase == f"{method}_stl_refit":
            winner_architecture = comp.get("stl_architecture")
            winner_metrics = comp.get("stl_test_metrics", {})
            break

    if winner_architecture is None:
        winner_architecture = winner.get("search_architecture") or winner.get("stl_architecture")

    return {
        "method": "reference_goliath",
        "reference_run_root": reference_summary.get("run_root"),
        "search_best_val": float(winner_value),
        "stl_best_val": float(winner_value),
        "winner": winner_kind or "reference",
        "winner_phase": winner_phase,
        "winner_value": float(winner_value),
        "search_architecture": winner_architecture,
        "stl_architecture": winner_architecture,
        "search_test_metrics": winner_metrics,
        "stl_test_metrics": winner_metrics,
        "reference": True,
    }


def build_train_cfg(suite_cfg: SuiteConfig) -> RunConfig:
    return RunConfig(
        data_dir=suite_cfg.data_dir,
        results_dir=suite_cfg.results_dir,
        run_root=suite_cfg.run_root,
        tasks=suite_cfg.tasks,
        phases=suite_cfg.methods,
        batch_size=suite_cfg.batch_size,
        num_workers=suite_cfg.num_workers,
        seed=suite_cfg.seed,
        stl_width=128,
        stl_depth=2,
        alt_start_width=1,
        alt_start_depth=1,
        patience=suite_cfg.patience,
        width_expansion_patience=10,
        depth_expansion_patience=5,
        delta=suite_cfg.delta,
        max_epochs=suite_cfg.max_epochs,
        lr=suite_cfg.lr,
        weight_decay=suite_cfg.weight_decay,
        grad_clip=suite_cfg.grad_clip,
        max_width=suite_cfg.max_width,
        max_depth=suite_cfg.max_depth,
        max_neurons=suite_cfg.max_neurons,
        width_stage_margin_patience=suite_cfg.width_stage_margin_patience,
        width_stage_min_improve_pct=suite_cfg.width_stage_min_improve_pct,
        use_bn=suite_cfg.use_bn,
        demo=False,
    )


def normalize_task_specific_widths(task: Task, cfg: SuiteConfig) -> List[int]:
    widths = set()
    widths.update(int(w) for w in BASE_WIDTH_CANDIDATES if int(w) <= int(cfg.max_width))
    widths.update(int(w) for w in [task.in_dim, task.in_dim * 2, task.in_dim * 4] if 2 <= int(w) <= int(cfg.max_width))
    widths.update(int(w) for w in [task.out_dim, task.out_dim * 2] if 2 <= int(w) <= int(cfg.max_width))
    widths.add(2)
    return sorted(widths)


def normalize_depth_candidates(cfg: SuiteConfig) -> List[int]:
    return [d for d in BASE_DEPTH_CANDIDATES if d <= int(cfg.max_depth)]


def architecture_pool(task: Task, cfg: SuiteConfig) -> List[List[int]]:
    widths = normalize_task_specific_widths(task, cfg)
    depths = normalize_depth_candidates(cfg)
    pool: List[List[int]] = []
    for depth in depths:
        for width in widths:
            hidden = [int(width)] * int(depth)
            if sum(hidden) + task.out_dim <= int(cfg.max_neurons):
                pool.append(hidden)
    # Deterministic order for grid search
    return pool


def task_seed_offset(task_name: str) -> int:
    return sum(ord(ch) for ch in task_name)


def architecture_key(hidden_widths: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(w) for w in hidden_widths)


def architecture_features(hidden_widths: Sequence[int], cfg: SuiteConfig) -> np.ndarray:
    width = float(max(hidden_widths)) if hidden_widths else 0.0
    depth = float(len(hidden_widths))
    return np.asarray(
        [
            depth / max(float(cfg.max_depth), 1.0),
            width / max(float(cfg.max_width), 1.0),
            math.log1p(width) / max(math.log1p(float(cfg.max_width)), 1e-6),
        ],
        dtype=np.float64,
    )


def make_candidate_model(task: Task, hidden_widths: Sequence[int], use_bn: bool, reconstruct: bool) -> MLP:
    if reconstruct:
        return make_reconstruction_model(task, hidden_widths, use_bn)
    return make_stl_model(task, hidden_widths, use_bn)


def train_candidate(
    *,
    task: Task,
    task_root: Path,
    method: str,
    candidate_index: int,
    hidden_widths: Sequence[int],
    cfg: RunConfig,
    train_cfg: RunConfig,
    device,
    reconstruct: bool,
    batch_controller: Optional[AdaptiveBatchController],
) -> Dict[str, Any]:
    method_root = method_root_for(task_root, method)
    candidate_dir = candidate_root_for(method_root, candidate_index, hidden_widths)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    existing_summary = load_completed_candidate_summary(candidate_dir)
    if existing_summary is not None:
        return existing_summary

    state_path = candidate_state_path(candidate_dir)
    if state_path.exists() and bool(read_json(state_path).get("completed", False)):
        model, meta, ckpt = load_candidate_model(candidate_dir, device)
        test_metrics = eval_final(model, task, device, reconstruct=reconstruct)
        summary = {
            "method": method,
            "candidate_dir": str(candidate_dir),
            "architecture": {"hidden_widths": [int(w) for w in hidden_widths], "in_dim": task.in_dim, "out_dim": task.out_dim, "use_bn": train_cfg.use_bn},
            "best_val": float(ckpt["best_val"]),
            "best_epoch": int(ckpt["best_epoch"]),
            "final_epoch": int(ckpt.get("epoch", ckpt["best_epoch"])),
            "best_checkpoint": str(candidate_dir / "checkpoint_best.pt"),
            "last_checkpoint": str(candidate_dir / "checkpoint_last.pt"),
            "test_metrics": test_metrics,
            "reconstruct": reconstruct,
        }
        write_json(candidate_summary_path(candidate_dir), summary)
        return summary

    model = make_candidate_model(task, hidden_widths, train_cfg.use_bn, reconstruct).to(device)
    logger = ContinuousLogger(candidate_dir, f"{task.name}_{method}", method)
    write_json(
        candidate_dir / "metadata.json",
        phase_metadata(
            task=task,
            phase_name=method,
            phase_kind=method,
            reconstruct=reconstruct,
            model=model,
            cfg=train_cfg,
            candidate_index=candidate_index,
            extra={"hidden_widths": [int(w) for w in hidden_widths], "search_method": method},
        ),
    )
    result = training_loop(
        task=task,
        model=model,
        candidate_dir=candidate_dir,
        cfg=train_cfg,
        device=device,
        logger=logger,
        reconstruct=reconstruct,
        resume=True,
        batch_controller=batch_controller,
    )
    logger.close()
    test_metrics = eval_final(model, task, device, reconstruct=reconstruct)
    summary = {
        "method": method,
        "candidate_dir": str(candidate_dir),
        "architecture": {"hidden_widths": [int(w) for w in hidden_widths], "in_dim": task.in_dim, "out_dim": task.out_dim, "use_bn": train_cfg.use_bn},
        "best_val": float(result.best_val),
        "best_epoch": int(result.best_epoch),
        "final_epoch": int(result.final_epoch),
        "best_checkpoint": str(result.best_checkpoint),
        "last_checkpoint": str(result.last_checkpoint),
        "test_metrics": test_metrics,
        "reconstruct": reconstruct,
    }
    write_json(candidate_dir / "candidate_summary.json", summary)
    return summary


def refit_stl_on_architecture(
    *,
    task: Task,
    task_root: Path,
    method: str,
    candidate_index: int,
    hidden_widths: Sequence[int],
    train_cfg: RunConfig,
    device,
    batch_controller: Optional[AdaptiveBatchController],
) -> Dict[str, Any]:
    return train_candidate(
        task=task,
        task_root=task_root,
        method=f"{method}_stl_refit",
        candidate_index=candidate_index,
        hidden_widths=hidden_widths,
        cfg=train_cfg,
        train_cfg=train_cfg,
        device=device,
        reconstruct=False,
        batch_controller=batch_controller,
    )


def run_grid_search(task: Task, task_root: Path, cfg: SuiteConfig, train_cfg: RunConfig, device, batch_controller) -> Dict[str, Any]:
    pool = architecture_pool(task, cfg)
    budget = len(pool) if int(cfg.candidate_budget) <= 0 else min(int(cfg.candidate_budget), len(pool))
    pool = pool[:budget]
    method = "grid"
    method_root = method_root_for(task_root, method)
    method_root.mkdir(parents=True, exist_ok=True)
    existing = load_json_if_exists(method_summary_path(task_root, method))
    if existing is not None and bool(existing.get("completed", False)):
        return existing

    state = load_method_state(task_root, method) or {
        "method": method,
        "task": task.name,
        "candidate_order": [list(map(int, arch)) for arch in pool],
        "budget": budget,
        "next_candidate_index": 0,
        "completed": False,
        "best_candidate_dir": None,
        "best_val": None,
    }

    reconstruct = task_reconstruct(task)
    candidates = collect_completed_candidate_summaries(method_root, task, device, reconstruct)
    start_index = int(state.get("next_candidate_index", 0))
    for i, arch in enumerate(pool[start_index:], start=start_index):
        state.update(
            {
                "active_candidate_index": i,
                "active_architecture": [int(w) for w in arch],
                "next_candidate_index": i,
                "completed": False,
            }
        )
        save_method_state(task_root, method, state)
        result = train_candidate(
            task=task,
            task_root=task_root,
            method=method,
            candidate_index=i,
            hidden_widths=arch,
            cfg=cfg,
            train_cfg=train_cfg,
            device=device,
            reconstruct=reconstruct,
            batch_controller=batch_controller,
        )
        candidates.append(result)
        state.update(
            {
                "active_candidate_index": None,
                "active_architecture": None,
                "next_candidate_index": i + 1,
                "best_candidate_dir": min(candidates, key=lambda x: float(x["best_val"]))["candidate_dir"],
                "best_val": min(float(x["best_val"]) for x in candidates),
            }
        )
        save_method_state(task_root, method, state)

    candidates = collect_completed_candidate_summaries(method_root, task, device, reconstruct)
    if not candidates:
        raise ValueError(f"No completed candidates found for {method} on task {task.name}")
    best = min(candidates, key=lambda x: float(x["best_val"]))
    refit = refit_stl_on_architecture(
        task=task,
        task_root=task_root,
        method=method,
        candidate_index=0,
        hidden_widths=best["architecture"]["hidden_widths"],
        train_cfg=train_cfg,
        device=device,
        batch_controller=batch_controller,
    )
    result = {"method": method, "search_candidates": candidates, "best": best, "stl_refit": refit, "completed": True}
    write_json(method_summary_path(task_root, method), result)
    state.update(
        {
            "completed": True,
            "best_candidate_dir": best["candidate_dir"],
            "best_val": float(best["best_val"]),
            "stl_refit_candidate_dir": refit["candidate_dir"],
        }
    )
    save_method_state(task_root, method, state)
    return result


def run_random_search(task: Task, task_root: Path, cfg: SuiteConfig, train_cfg: RunConfig, device, batch_controller) -> Dict[str, Any]:
    pool = architecture_pool(task, cfg)
    rng = random.Random(int(cfg.seed) + task_seed_offset(task.name) + 17)
    budget = len(pool) if int(cfg.candidate_budget) <= 0 else min(int(cfg.candidate_budget), len(pool))
    pool = pool[:]
    rng.shuffle(pool)
    pool = pool[:budget]
    method = "random"
    method_root = method_root_for(task_root, method)
    method_root.mkdir(parents=True, exist_ok=True)
    existing = load_json_if_exists(method_summary_path(task_root, method))
    if existing is not None and bool(existing.get("completed", False)):
        return existing

    state = load_method_state(task_root, method) or {
        "method": method,
        "task": task.name,
        "candidate_order": [list(map(int, arch)) for arch in pool],
        "budget": budget,
        "next_candidate_index": 0,
        "completed": False,
        "best_candidate_dir": None,
        "best_val": None,
    }

    reconstruct = task_reconstruct(task)
    candidates = collect_completed_candidate_summaries(method_root, task, device, reconstruct)
    start_index = int(state.get("next_candidate_index", 0))
    for i, arch in enumerate(pool[start_index:], start=start_index):
        state.update(
            {
                "active_candidate_index": i,
                "active_architecture": [int(w) for w in arch],
                "next_candidate_index": i,
                "completed": False,
            }
        )
        save_method_state(task_root, method, state)
        result = train_candidate(
            task=task,
            task_root=task_root,
            method=method,
            candidate_index=i,
            hidden_widths=arch,
            cfg=cfg,
            train_cfg=train_cfg,
            device=device,
            reconstruct=reconstruct,
            batch_controller=batch_controller,
        )
        candidates.append(result)
        state.update(
            {
                "active_candidate_index": None,
                "active_architecture": None,
                "next_candidate_index": i + 1,
                "best_candidate_dir": min(candidates, key=lambda x: float(x["best_val"]))["candidate_dir"],
                "best_val": min(float(x["best_val"]) for x in candidates),
            }
        )
        save_method_state(task_root, method, state)

    candidates = collect_completed_candidate_summaries(method_root, task, device, reconstruct)
    if not candidates:
        raise ValueError(f"No completed candidates found for {method} on task {task.name}")
    best = min(candidates, key=lambda x: float(x["best_val"]))
    refit = refit_stl_on_architecture(
        task=task,
        task_root=task_root,
        method=method,
        candidate_index=0,
        hidden_widths=best["architecture"]["hidden_widths"],
        train_cfg=train_cfg,
        device=device,
        batch_controller=batch_controller,
    )
    result = {"method": method, "search_candidates": candidates, "best": best, "stl_refit": refit, "completed": True}
    write_json(method_summary_path(task_root, method), result)
    state.update(
        {
            "completed": True,
            "best_candidate_dir": best["candidate_dir"],
            "best_val": float(best["best_val"]),
            "stl_refit_candidate_dir": refit["candidate_dir"],
        }
    )
    save_method_state(task_root, method, state)
    return result


def run_bayesian_search(task: Task, task_root: Path, cfg: SuiteConfig, train_cfg: RunConfig, device, batch_controller) -> Dict[str, Any]:
    pool = architecture_pool(task, cfg)
    budget = len(pool) if int(cfg.candidate_budget) <= 0 else min(int(cfg.candidate_budget), len(pool))
    if budget <= 0:
        raise ValueError("candidate_budget must be positive")
    rng = random.Random(int(cfg.seed) + task_seed_offset(task.name) + 33)
    method = "bayes"
    method_root = method_root_for(task_root, method)
    method_root.mkdir(parents=True, exist_ok=True)

    existing = load_json_if_exists(method_summary_path(task_root, method))
    if existing is not None and bool(existing.get("completed", False)):
        return existing

    state = load_method_state(task_root, method)
    if state is None:
        pending = [list(map(int, arch)) for arch in pool]
        rng.shuffle(pending)
        observed: List[Dict[str, Any]] = []
        init_count = min(max(3, int(cfg.bayes_warmup)), budget, len(pending))
        state = {
            "method": method,
            "task": task.name,
            "budget": budget,
            "pending": pending,
            "observed": observed,
            "candidate_index": 0,
            "init_count": init_count,
            "completed": False,
            "best_candidate_dir": None,
            "best_val": None,
        }
    else:
        pending = [list(map(int, arch)) for arch in state.get("pending", [])]
        observed = [dict(item) for item in state.get("observed", [])]
        init_count = int(state.get("init_count", min(max(3, int(cfg.bayes_warmup)), budget, len(pending))))
        # Reconcile from on-disk candidates if state was not fully persisted.
        if not observed:
            completed = collect_completed_candidate_summaries(method_root, task, device, reconstruct)
            observed = [
                {
                    "architecture": summary["architecture"]["hidden_widths"],
                    "best_val": summary["best_val"],
                    "candidate_dir": summary["candidate_dir"],
                    "candidate_index": int(Path(summary["candidate_dir"]).name.split("_")[1]),
                    "summary": summary,
                }
                for summary in completed
            ]
            observed_keys = {architecture_key(item["architecture"]) for item in observed}
            pending = [arch for arch in pending if architecture_key(arch) not in observed_keys]
            state["observed"] = observed
            state["pending"] = pending
            save_method_state(task_root, method, state)

    candidates: List[Dict[str, Any]] = [item.get("summary") for item in observed if item.get("summary") is not None]
    next_index = int(state.get("candidate_index", len(candidates)))
    if next_index < len(candidates):
        next_index = len(candidates)

    for i in range(min(init_count, len(pending))):
        arch = pending.pop(0)
        state.update(
            {
                "candidate_index": next_index,
                "active_architecture": arch,
                "pending": pending,
                "observed": observed,
                "completed": False,
            }
        )
        save_method_state(task_root, method, state)
        res = train_candidate(
            task=task,
            task_root=task_root,
            method=method,
            candidate_index=next_index,
            hidden_widths=arch,
            cfg=cfg,
            train_cfg=train_cfg,
            device=device,
            reconstruct=reconstruct,
            batch_controller=batch_controller,
        )
        candidates.append(res)
        observed.append(
            {
                "architecture": res["architecture"]["hidden_widths"],
                "best_val": res["best_val"],
                "candidate_dir": res["candidate_dir"],
                "candidate_index": next_index,
                "summary": res,
            }
        )
        next_index += 1
        state.update(
            {
                "candidate_index": next_index,
                "active_architecture": None,
                "pending": pending,
                "observed": observed,
            }
        )
        save_method_state(task_root, method, state)

    def fit_gp() -> Optional[GaussianProcessRegressor]:
        if len(observed) < 2:
            return None
        xs = np.vstack([architecture_features(res["architecture"], cfg) for res in observed])
        ys = np.asarray([float(res["best_val"]) for res in observed], dtype=np.float64)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(xs.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=int(cfg.seed))
        gp.fit(xs, ys)
        return gp

    while len(candidates) < budget and pending:
        gp = fit_gp()
        if gp is None:
            arch = pending.pop(0)
        else:
            xs_pending = np.vstack([architecture_features(arch, cfg) for arch in pending])
            mu, sigma = gp.predict(xs_pending, return_std=True)
            best_y = min(float(res["best_val"]) for res in observed)
            imp = best_y - mu - 1e-6
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma > 0)
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            choice = int(np.argmax(ei))
            arch = pending.pop(choice)
        state.update(
            {
                "candidate_index": next_index,
                "active_architecture": arch,
                "pending": pending,
                "observed": observed,
                "completed": False,
            }
        )
        save_method_state(task_root, method, state)
        res = train_candidate(
            task=task,
            task_root=task_root,
            method=method,
            candidate_index=next_index,
            hidden_widths=arch,
            cfg=cfg,
            train_cfg=train_cfg,
            device=device,
            reconstruct=reconstruct,
            batch_controller=batch_controller,
        )
        candidates.append(res)
        observed.append(
            {
                "architecture": res["architecture"]["hidden_widths"],
                "best_val": res["best_val"],
                "candidate_dir": res["candidate_dir"],
                "candidate_index": next_index,
                "summary": res,
            }
        )
        next_index += 1
        state.update(
            {
                "candidate_index": next_index,
                "active_architecture": None,
                "pending": pending,
                "observed": observed,
            }
        )
        save_method_state(task_root, method, state)

    candidates = collect_completed_candidate_summaries(method_root, task, device, reconstruct)
    if not candidates:
        raise ValueError(f"No completed candidates found for {method} on task {task.name}")
    best = min(candidates, key=lambda x: float(x["best_val"]))
    refit = refit_stl_on_architecture(
        task=task,
        task_root=task_root,
        method=method,
        candidate_index=0,
        hidden_widths=best["architecture"]["hidden_widths"],
        train_cfg=train_cfg,
        device=device,
        batch_controller=batch_controller,
    )
    result = {"method": method, "search_candidates": candidates, "best": best, "stl_refit": refit, "completed": True}
    write_json(method_summary_path(task_root, method), result)
    state.update(
        {
            "completed": True,
            "best_candidate_dir": best["candidate_dir"],
            "best_val": float(best["best_val"]),
            "stl_refit_candidate_dir": refit["candidate_dir"],
            "pending": pending,
            "observed": observed,
        }
    )
    save_method_state(task_root, method, state)
    return result


def run_nas_search(task: Task, task_root: Path, cfg: SuiteConfig, train_cfg: RunConfig, device, batch_controller) -> Dict[str, Any]:
    method = "nas"
    method_root = method_root_for(task_root, method)
    method_root.mkdir(parents=True, exist_ok=True)
    budget = len(architecture_pool(task, cfg)) if int(cfg.candidate_budget) <= 0 else int(cfg.candidate_budget)
    existing = load_json_if_exists(method_summary_path(task_root, method))
    if existing is not None and bool(existing.get("completed", False)):
        return existing

    state = load_method_state(task_root, method) or {
        "method": method,
        "task": task.name,
        "budget": budget,
        "current_arch": [2, 2],
        "best_architecture": [2, 2],
        "best_val": None,
        "candidate_index": 0,
        "no_improve": 0,
        "pending_children": [],
        "completed": False,
    }

    current_arch = [int(w) for w in state.get("current_arch", [2, 2])]
    best_architecture = [int(w) for w in state.get("best_architecture", current_arch)]
    best_val = state.get("best_val")
    no_improve = int(state.get("no_improve", 0))
    candidate_index = int(state.get("candidate_index", 0))
    pending_children: List[List[int]] = [list(map(int, child)) for child in state.get("pending_children", [])]

    reconstruct = task_reconstruct(task)
    candidates: List[Dict[str, Any]] = collect_completed_candidate_summaries(method_root, task, device, reconstruct)

    def evaluate(arch: Sequence[int]) -> Dict[str, Any]:
        nonlocal candidate_index
        state.update(
            {
                "candidate_index": candidate_index,
                "current_arch": [int(w) for w in current_arch],
                "best_architecture": best_architecture,
                "best_val": best_val,
                "no_improve": no_improve,
                "pending_children": pending_children,
                "active_architecture": [int(w) for w in arch],
                "completed": False,
            }
        )
        save_method_state(task_root, method, state)
        res = train_candidate(
            task=task,
            task_root=task_root,
            method=method,
            candidate_index=candidate_index,
            hidden_widths=arch,
            cfg=cfg,
            train_cfg=train_cfg,
            device=device,
            reconstruct=reconstruct,
            batch_controller=batch_controller,
        )
        candidate_index += 1
        return res

    if not candidates:
        first = evaluate(current_arch)
        candidates.append(first)
        best_val = float(first["best_val"])
        best_architecture = list(first["architecture"]["hidden_widths"])
        state.update(
            {
                "current_arch": current_arch,
                "best_architecture": best_architecture,
                "best_val": best_val,
                "candidate_index": candidate_index,
                "no_improve": no_improve,
                "pending_children": pending_children,
            }
        )
        save_method_state(task_root, method, state)

    while candidate_index < budget and no_improve < int(cfg.nas_patience):
        if not pending_children:
            children: List[List[int]] = []
            widen = expand_width(make_candidate_model(task, current_arch, train_cfg.use_bn, reconstruct), 1, int(cfg.max_width))
            deepen = expand_depth(make_candidate_model(task, current_arch, train_cfg.use_bn, reconstruct), int(cfg.max_depth))
            if widen is not None:
                children.append(list(widen.hidden_widths))
            if deepen is not None:
                children.append(list(deepen.hidden_widths))
            if not children:
                break
            pending_children = children
            state["pending_children"] = pending_children
            save_method_state(task_root, method, state)

        child_results = []
        for child in list(pending_children):
            if candidate_index >= budget:
                break
            child_results.append(evaluate(child))
        pending_children = []
        state["pending_children"] = pending_children
        save_method_state(task_root, method, state)
        if not child_results:
            break
        child_best = min(child_results, key=lambda x: float(x["best_val"]))
        if best_val is None or float(child_best["best_val"]) < float(best_val) - float(cfg.delta):
            best_val = float(child_best["best_val"])
            best_architecture = list(child_best["architecture"]["hidden_widths"])
            current_arch = best_architecture[:]
            no_improve = 0
        else:
            no_improve += 1
            current_arch = list(child_best["architecture"]["hidden_widths"])
        state.update(
            {
                "current_arch": current_arch,
                "best_architecture": best_architecture,
                "best_val": best_val,
                "candidate_index": candidate_index,
                "no_improve": no_improve,
                "pending_children": pending_children,
                "completed": False,
            }
        )
        save_method_state(task_root, method, state)

    candidates = collect_completed_candidate_summaries(method_root, task, device, reconstruct)
    if not candidates:
        raise ValueError(f"No completed candidates found for {method} on task {task.name}")
    best = min(candidates, key=lambda x: float(x["best_val"]))
    refit = refit_stl_on_architecture(
        task=task,
        task_root=task_root,
        method=method,
        candidate_index=0,
        hidden_widths=best["architecture"]["hidden_widths"],
        train_cfg=train_cfg,
        device=device,
        batch_controller=batch_controller,
    )
    result = {"method": method, "search_candidates": candidates, "best": best, "stl_refit": refit, "completed": True}
    write_json(method_summary_path(task_root, method), result)
    state.update(
        {
            "completed": True,
            "current_arch": current_arch,
            "best_architecture": best["architecture"]["hidden_widths"],
            "best_val": float(best["best_val"]),
            "candidate_index": candidate_index,
            "pending_children": [],
            "stl_refit_candidate_dir": refit["candidate_dir"],
        }
    )
    save_method_state(task_root, method, state)
    return result


def run_adp_search_method(
    task: Task,
    task_root: Path,
    cfg: SuiteConfig,
    train_cfg: RunConfig,
    device,
    batch_controller,
    method: str,
    budget: int,
) -> Dict[str, Any]:
    reconstruct = task_reconstruct(task)
    adp_cfg = ADPConfig(
        adp_mode=adp_mode_for_method(method),
        delta=float(cfg.delta),
        patience=int(cfg.patience),
        trials_width=0,
        trials_depth=0,
        ex_k=1,
        max_width=int(cfg.max_width),
        max_depth=int(cfg.max_depth),
        max_neurons=int(cfg.max_neurons),
        width_stage_margin_patience=int(cfg.width_stage_margin_patience),
        width_stage_min_improve_pct=float(cfg.width_stage_min_improve_pct),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        grad_clip=float(cfg.grad_clip),
        max_epochs=int(cfg.max_epochs),
        metrics_interval=5,
    )
    model = make_candidate_model(task, [1], train_cfg.use_bn, reconstruct).to(device)
    logger = ContinuousLogger(method_root_for(task_root, method), f"{task.name}_{method}", method)
    best_val, best_model = adp_search(
        model,
        task,
        adp_cfg,
        device,
        logger=logger,
        batch_controller=batch_controller,
        max_candidates=int(budget) if int(budget) > 0 else None,
    )
    logger.close()
    best_arch = list(best_model.hidden_widths)
    best_summary = {
        "method": method,
        "architecture": {"hidden_widths": best_arch, "in_dim": task.in_dim, "out_dim": task.out_dim, "use_bn": train_cfg.use_bn},
        "best_val": float(best_val),
        "best_epoch": None,
        "final_epoch": None,
        "best_checkpoint": None,
        "last_checkpoint": None,
        "test_metrics": eval_final(best_model, task, device, reconstruct=reconstruct),
        "reconstruct": reconstruct,
    }
    method_root = method_root_for(task_root, method)
    write_json(method_root / "method_summary.json", best_summary)
    refit = refit_stl_on_architecture(
        task=task,
        task_root=task_root,
        method=method,
        candidate_index=0,
        hidden_widths=best_arch,
        train_cfg=train_cfg,
        device=device,
        batch_controller=batch_controller,
    )
    return {"method": method, "search_candidates": [best_summary], "best": best_summary, "stl_refit": refit}


def method_comparison(method_result: Dict[str, Any], task: Task) -> Dict[str, Any]:
    best = method_result["best"]
    refit = method_result["stl_refit"]
    best_val = float(best["best_val"])
    stl_val = float(refit["best_val"])
    winner = "search" if best_val <= stl_val else "stl"
    return {
        "method": method_result["method"],
        "search_best_val": best_val,
        "stl_best_val": stl_val,
        "winner": winner,
        "winner_phase": method_result["method"] if winner == "search" else f"{method_result['method']}_stl_refit",
        "winner_value": min(best_val, stl_val),
        "search_architecture": best["architecture"],
        "stl_architecture": refit["architecture"],
        "search_test_metrics": best.get("test_metrics", {}),
        "stl_test_metrics": refit.get("test_metrics", {}),
    }


def run_task_suite(task: Task, task_root: Path, cfg: SuiteConfig, train_cfg: RunConfig, device, batch_controller) -> Dict[str, Any]:
    existing_task_summary = load_json_if_exists(task_root / "task_summary.json") or {}
    task_summary: Dict[str, Any] = {
        "task": task.name,
        "reference": existing_task_summary.get("reference"),
        "method_runs": list(existing_task_summary.get("method_runs", [])),
        "comparisons": list(existing_task_summary.get("comparisons", [])),
        "winner": existing_task_summary.get("winner"),
    }

    reference_summary = load_reference_task_summary(cfg.reference_run_root, task.name)
    if reference_summary is not None:
        task_summary["reference"] = reference_summary
        reference_comparison = reference_winner_comparison(reference_summary)
    else:
        reference_comparison = None

    best_overall: Optional[Dict[str, Any]] = task_summary.get("winner") or reference_comparison
    effective_budget = len(architecture_pool(task, cfg)) if int(cfg.candidate_budget) <= 0 else int(cfg.candidate_budget)

    if reference_comparison is not None:
        if not any(comp.get("reference", False) for comp in task_summary["comparisons"]):
            task_summary["comparisons"].append(reference_comparison)

    completed_methods = {str(entry.get("method")) for entry in task_summary.get("method_runs", []) if entry.get("method")}
    task_state = load_task_state(task_root) or {
        "task": task.name,
        "methods": list(cfg.methods),
        "next_method_index": 0,
        "completed_methods": sorted(completed_methods),
        "winner": task_summary.get("winner"),
        "completed": False,
    }

    for method_index, method in enumerate(cfg.methods):
        if method in ADP_METHODS:
            raise ValueError(
                f"run_search_suite.py is baseline-only and does not run ADP methods. "
                f"Use run_goliath.py for {method}."
            )
        if method in completed_methods:
            continue
        method_root_for(task_root, method).mkdir(parents=True, exist_ok=True)
        existing_method_summary = load_json_if_exists(method_summary_path(task_root, method))
        if existing_method_summary is not None and bool(existing_method_summary.get("completed", False)):
            result = existing_method_summary
        else:
            task_state.update(
                {
                    "next_method_index": method_index,
                    "active_method": method,
                    "completed": False,
                }
            )
            save_task_state(task_root, task_state)
            existing_method_summary = load_json_if_exists(method_summary_path(task_root, method))
            if existing_method_summary is not None and bool(existing_method_summary.get("completed", False)):
                result = existing_method_summary
            elif method == "grid":
                result = run_grid_search(task, task_root, cfg, train_cfg, device, batch_controller)
            elif method == "random":
                result = run_random_search(task, task_root, cfg, train_cfg, device, batch_controller)
            elif method == "bayes":
                result = run_bayesian_search(task, task_root, cfg, train_cfg, device, batch_controller)
            elif method == "nas":
                result = run_nas_search(task, task_root, cfg, train_cfg, device, batch_controller)
            elif method in ADP_METHODS:
                result = run_adp_search_method(task, task_root, cfg, train_cfg, device, batch_controller, method, effective_budget)
            else:
                raise ValueError(f"Unknown method: {method}")

        comparison = method_comparison(result, task)
        task_summary["method_runs"].append(result)
        task_summary["comparisons"].append(comparison)
        if best_overall is None or float(comparison["winner_value"]) < float(best_overall["winner_value"]):
            best_overall = comparison
        task_summary["winner"] = best_overall
        write_json(task_root / "task_summary.json", task_summary)
        task_state.update(
            {
                "next_method_index": method_index + 1,
                "completed_methods": sorted({str(entry.get("method")) for entry in task_summary.get("method_runs", []) if entry.get("method")}),
                "winner": task_summary["winner"],
                "completed": False,
            }
        )
        save_task_state(task_root, task_state)

    task_summary["winner"] = best_overall
    write_json(task_root / "task_summary.json", task_summary)
    task_state.update(
        {
            "next_method_index": len(cfg.methods),
            "completed_methods": sorted({str(entry.get("method")) for entry in task_summary.get("method_runs", []) if entry.get("method")}),
            "winner": task_summary["winner"],
            "completed": True,
        }
    )
    save_task_state(task_root, task_state)
    return task_summary


def render_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# DAE/DNN Search Suite Report")
    lines.append("")
    lines.append(f"- Run root: `{report.get('run_root', 'n/a')}`")
    lines.append(f"- Git commit: `{report.get('git_commit', 'n/a')}`")
    lines.append(f"- Tasks completed: `{report.get('summary', {}).get('tasks_completed', 0)}`")
    lines.append("")
    for task_report in report.get("tasks", []):
        lines.append(f"## Task: {task_report.get('task', 'n/a')}")
        reference = task_report.get("reference")
        if reference:
            ref_winner = reference.get("winner") or {}
            lines.append(
                f"- Reference goliath winner: `{ref_winner.get('winner', 'n/a')}` via `{ref_winner.get('winner_phase', 'n/a')}` at `{ref_winner.get('winner_value', 'n/a')}`"
            )
        winner = task_report.get("winner") or {}
        lines.append(
            f"- Overall winner: `{winner.get('winner', 'n/a')}` via `{winner.get('winner_phase', 'n/a')}` at `{winner.get('winner_value', 'n/a')}`"
        )
        lines.append("")
        lines.append("| Method | Search arch | Search val | STL arch | STL val | Winner |")
        lines.append("|---|---|---:|---|---:|---|")
        for comp in task_report.get("comparisons", []):
            lines.append(
                "| "
                f"{comp.get('method', 'n/a')} | "
                f"{format_architecture_for_report(comp.get('search_architecture'))} | "
                f"{comp.get('search_best_val', 'n/a')} | "
                f"{format_architecture_for_report(comp.get('stl_architecture'))} | "
                f"{comp.get('stl_best_val', 'n/a')} | "
                f"{comp.get('winner', 'n/a')} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_summary_rows(task_name: str, task_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for comp in task_summary.get("comparisons", []):
        search_metrics = comp.get("search_test_metrics") or {}
        stl_metrics = comp.get("stl_test_metrics") or {}
        rows.append(
            {
                "task": task_name,
                "method": comp.get("method"),
                "reference": bool(comp.get("reference", False)),
                "search_best_val": comp.get("search_best_val"),
                "stl_best_val": comp.get("stl_best_val"),
                "winner": comp.get("winner"),
                "winner_phase": comp.get("winner_phase"),
                "winner_value": comp.get("winner_value"),
                "search_architecture": format_architecture_for_report(comp.get("search_architecture")),
                "stl_architecture": format_architecture_for_report(comp.get("stl_architecture")),
                "search_test_loss": search_metrics.get("test_loss"),
                "search_test_acc": search_metrics.get("test_acc"),
                "search_knn_acc": search_metrics.get("knn_acc"),
                "search_cluster_nmi": search_metrics.get("cluster_nmi"),
                "stl_test_loss": stl_metrics.get("test_loss"),
                "stl_test_acc": stl_metrics.get("test_acc"),
                "stl_knn_acc": stl_metrics.get("knn_acc"),
                "stl_cluster_nmi": stl_metrics.get("cluster_nmi"),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="DAE/DNN exhaustive benchmark search suite")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="DAE/DNN/results")
    p.add_argument("--run-root", type=str, default=None)
    p.add_argument("--reference-run-root", type=str, default=None, help="Existing goliath run root to use as ADP/STL reference")
    p.add_argument("--tasks", type=str, nargs="+", default=["all"])
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_BASELINE_METHODS,
        help="Search methods to run. Defaults to the stronger baselines only.",
    )
    p.add_argument("--batch-size", type=int, default=32768)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--width-stage-margin-patience", type=int, default=10)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--no-bn", action="store_true")
    p.add_argument("--candidate-budget", type=int, default=0, help="Max candidate trainings per method; 0 means exhaustive grid size")
    p.add_argument("--bayes-warmup", type=int, default=5)
    p.add_argument("--nas-patience", type=int, default=5)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--demo-tasks", type=int, default=1)
    args = p.parse_args()

    tasks = task_names() if "all" in [t.lower() for t in args.tasks] else args.tasks
    if args.demo:
        tasks = tasks[: max(1, int(args.demo_tasks))]
        if args.max_epochs > 1:
            args.max_epochs = 1
        if args.patience > 1:
            args.patience = 1

    cfg = SuiteConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        run_root=args.run_root,
        tasks=tasks,
        methods=args.methods,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        patience=int(args.patience),
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
        use_bn=not bool(args.no_bn),
        candidate_budget=int(args.candidate_budget),
        bayes_warmup=int(args.bayes_warmup),
        nas_patience=int(args.nas_patience),
        reference_run_root=args.reference_run_root,
    )
    if any(method in ADP_METHODS for method in cfg.methods):
        raise ValueError(
            "run_search_suite.py is baseline-only. ADP methods are not allowed here; "
            "use run_goliath.py for ADP/STL comparisons."
        )
    train_cfg = build_train_cfg(cfg)

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_root = build_run_root(train_cfg)
    run_root.mkdir(parents=True, exist_ok=True)

    write_json(
        run_root / "run_metadata.json",
        {
            "config": asdict(cfg),
            "train_config": asdict(train_cfg),
            "git_commit": git_commit(),
            "device": str(device),
            "tasks": tasks,
            "available_tasks": task_names(),
            "timestamp": now_stamp(),
            "methods": cfg.methods,
            "reference_run_root": cfg.reference_run_root,
        },
    )

    progress_path = run_root / "run_progress.csv"
    task_summaries: Dict[str, Dict[str, Any]] = {}
    task_batch_controllers: Dict[str, AdaptiveBatchController] = {}
    task_objects: Dict[str, Task] = {}

    for task_name in tasks:
        batch_size = int(cfg.batch_size)
        task = build_task(task_name, cfg.data_dir, batch_size, cfg.num_workers, cfg.seed)
        refresh_task_loaders(task, batch_size)
        task_objects[task_name] = task
        task_root = task_root_for(run_root, task_name)
        task_root.mkdir(parents=True, exist_ok=True)
        controller = AdaptiveBatchController(
            batch_size,
            threshold_gb=DEFAULT_VRAM_BUDGET_GB,
            poll_interval_sec=30.0,
            shrink_factor=0.75,
            state_path=task_root / "_batch_size_state.json",
        )
        controller.start()
        task_batch_controllers[task_name] = controller

    try:
        for task_name in tasks:
            task = task_objects[task_name]
            task_root = task_root_for(run_root, task_name)
            controller = task_batch_controllers[task_name]
            refresh_task_loaders(task, controller.current_batch_size)
            summary = run_task_suite(task, task_root, cfg, train_cfg, device, controller)
            task_summaries[task_name] = summary
            write_json(task_root / "task_summary.json", summary)
            rows = build_summary_rows(task_name, summary)
            if rows:
                write_csv(
                    task_root / "task_summary.csv",
                    rows,
                    [
                        "task",
                        "method",
                        "reference",
                        "search_best_val",
                        "stl_best_val",
                        "winner",
                        "winner_phase",
                        "winner_value",
                        "search_architecture",
                        "stl_architecture",
                        "search_test_loss",
                        "search_test_acc",
                        "search_knn_acc",
                        "search_cluster_nmi",
                        "stl_test_loss",
                        "stl_test_acc",
                        "stl_knn_acc",
                        "stl_cluster_nmi",
                    ],
                )
    finally:
        for controller in task_batch_controllers.values():
            controller.stop()

        final_report = {
            "run_root": str(run_root),
            "git_commit": git_commit(),
            "device": str(device),
            "config": asdict(cfg),
            "summary": {
                "tasks_requested": list(tasks),
                "tasks_completed": [name for name, summary in task_summaries.items() if summary.get("winner") is not None],
                "num_tasks_requested": len(tasks),
                "num_tasks_completed": sum(1 for summary in task_summaries.values() if summary.get("winner") is not None),
            },
            "tasks": [task_summaries[name] for name in tasks if name in task_summaries],
        }
        write_json(run_root / "final_report.json", final_report)
        write_text(run_root / "final_report.md", render_report(final_report))


if __name__ == "__main__":
    main()
    reconstruct = task_reconstruct(task)
