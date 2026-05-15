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
    "adp_width_only",
    "adp_depth_only",
    "adp_width_to_depth",
    "adp_depth_to_width",
    "adp_alt_width",
    "adp_alt_depth",
]
SEARCH_METHODS = ["grid", "random", "bayes", "nas"] + ADP_METHODS


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
    use_bn: bool
    candidate_budget: int
    bayes_warmup: int
    nas_patience: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_reconstruct(task: Task) -> bool:
    return task.task_type == "reconstruction"


def is_adp_method(method: str) -> bool:
    return method.startswith("adp_")


def adp_mode_for_method(method: str) -> str:
    mapping = {
        "adp_width_only": "width_only",
        "adp_depth_only": "depth_only",
        "adp_width_to_depth": "width_to_depth",
        "adp_depth_to_width": "depth_to_width",
        "adp_alt_width": "alt_width",
        "adp_alt_depth": "alt_depth",
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


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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
        alt_start_width=2,
        alt_start_depth=2,
        patience=suite_cfg.patience,
        delta=suite_cfg.delta,
        max_epochs=suite_cfg.max_epochs,
        lr=suite_cfg.lr,
        weight_decay=suite_cfg.weight_decay,
        grad_clip=suite_cfg.grad_clip,
        max_width=suite_cfg.max_width,
        max_depth=suite_cfg.max_depth,
        max_neurons=suite_cfg.max_neurons,
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
    candidates = []
    for i, arch in enumerate(pool):
        candidates.append(
            train_candidate(
                task=task,
                task_root=task_root,
                method=method,
                candidate_index=i,
                hidden_widths=arch,
                cfg=cfg,
                train_cfg=train_cfg,
                device=device,
                reconstruct=True,
                batch_controller=batch_controller,
            )
        )
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
    return {"method": method, "search_candidates": candidates, "best": best, "stl_refit": refit}


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
    candidates = []
    for i, arch in enumerate(pool):
        candidates.append(
            train_candidate(
                task=task,
                task_root=task_root,
                method=method,
                candidate_index=i,
                hidden_widths=arch,
                cfg=cfg,
                train_cfg=train_cfg,
                device=device,
                reconstruct=True,
                batch_controller=batch_controller,
            )
        )
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
    return {"method": method, "search_candidates": candidates, "best": best, "stl_refit": refit}


def run_bayesian_search(task: Task, task_root: Path, cfg: SuiteConfig, train_cfg: RunConfig, device, batch_controller) -> Dict[str, Any]:
    pool = architecture_pool(task, cfg)
    budget = len(pool) if int(cfg.candidate_budget) <= 0 else min(int(cfg.candidate_budget), len(pool))
    if budget <= 0:
        raise ValueError("candidate_budget must be positive")
    rng = random.Random(int(cfg.seed) + task_seed_offset(task.name) + 33)
    method = "bayes"
    method_root = method_root_for(task_root, method)
    method_root.mkdir(parents=True, exist_ok=True)

    observed: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    pending = pool[:]
    rng.shuffle(pending)
    init_count = min(max(3, int(cfg.bayes_warmup)), budget, len(pending))

    candidates: List[Dict[str, Any]] = []
    for i in range(init_count):
        arch = pending.pop(0)
        res = train_candidate(
            task=task,
            task_root=task_root,
            method=method,
            candidate_index=i,
            hidden_widths=arch,
            cfg=cfg,
            train_cfg=train_cfg,
            device=device,
            reconstruct=True,
            batch_controller=batch_controller,
        )
        candidates.append(res)
        observed[architecture_key(arch)] = res

    def fit_gp() -> Optional[GaussianProcessRegressor]:
        if len(observed) < 2:
            return None
        xs = np.vstack([architecture_features(res["architecture"]["hidden_widths"], cfg) for res in observed.values()])
        ys = np.asarray([float(res["best_val"]) for res in observed.values()], dtype=np.float64)
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
            best_y = min(float(res["best_val"]) for res in observed.values())
            imp = best_y - mu - 1e-6
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma > 0)
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            choice = int(np.argmax(ei))
            arch = pending.pop(choice)
        res = train_candidate(
            task=task,
            task_root=task_root,
            method=method,
            candidate_index=len(candidates),
            hidden_widths=arch,
            cfg=cfg,
            train_cfg=train_cfg,
            device=device,
            reconstruct=True,
            batch_controller=batch_controller,
        )
        candidates.append(res)
        observed[architecture_key(arch)] = res

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
    return {"method": method, "search_candidates": candidates, "best": best, "stl_refit": refit}


def run_nas_search(task: Task, task_root: Path, cfg: SuiteConfig, train_cfg: RunConfig, device, batch_controller) -> Dict[str, Any]:
    method = "nas"
    method_root = method_root_for(task_root, method)
    method_root.mkdir(parents=True, exist_ok=True)
    budget = len(architecture_pool(task, cfg)) if int(cfg.candidate_budget) <= 0 else int(cfg.candidate_budget)
    candidates: List[Dict[str, Any]] = []
    current_arch = [2, 2]
    no_improve = 0
    candidate_index = 0

    def evaluate(arch: Sequence[int]) -> Dict[str, Any]:
        nonlocal candidate_index
        res = train_candidate(
            task=task,
            task_root=task_root,
            method=method,
            candidate_index=candidate_index,
            hidden_widths=arch,
            cfg=cfg,
            train_cfg=train_cfg,
            device=device,
            reconstruct=True,
            batch_controller=batch_controller,
        )
        candidate_index += 1
        return res

    candidates.append(evaluate(current_arch))
    best = candidates[-1]
    while candidate_index < budget and no_improve < int(cfg.nas_patience):
        children: List[List[int]] = []
        widen = expand_width(make_reconstruction_model(task, current_arch, train_cfg.use_bn), 1, int(cfg.max_width))
        deepen = expand_depth(make_reconstruction_model(task, current_arch, train_cfg.use_bn), int(cfg.max_depth))
        if widen is not None:
            children.append(list(widen.hidden_widths))
        if deepen is not None:
            children.append(list(deepen.hidden_widths))
        if not children:
            break
        child_results = [evaluate(child) for child in children if candidate_index < budget]
        candidates.extend(child_results)
        if not child_results:
            break
        child_best = min(child_results, key=lambda x: float(x["best_val"]))
        if float(child_best["best_val"]) < float(best["best_val"]) - float(cfg.delta):
            best = child_best
            current_arch = list(child_best["architecture"]["hidden_widths"])
            no_improve = 0
        else:
            no_improve += 1
            current_arch = list(child_best["architecture"]["hidden_widths"])

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
    return {"method": method, "search_candidates": candidates, "best": best, "stl_refit": refit}


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
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        grad_clip=float(cfg.grad_clip),
        max_epochs=int(cfg.max_epochs),
        metrics_interval=5,
    )
    model = make_reconstruction_model(task, [2, 2], train_cfg.use_bn).to(device)
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
        "test_metrics": eval_final(best_model, task, device, reconstruct=True),
        "reconstruct": True,
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
    task_summary: Dict[str, Any] = {
        "task": task.name,
        "method_runs": [],
        "comparisons": [],
        "winner": None,
    }

    best_overall: Optional[Dict[str, Any]] = None
    effective_budget = len(architecture_pool(task, cfg)) if int(cfg.candidate_budget) <= 0 else int(cfg.candidate_budget)

    for method in cfg.methods:
        method_root_for(task_root, method).mkdir(parents=True, exist_ok=True)
        if method == "grid":
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

    task_summary["winner"] = best_overall
    write_json(task_root / "task_summary.json", task_summary)
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
        rows.append(
            {
                "task": task_name,
                "method": comp.get("method"),
                "search_best_val": comp.get("search_best_val"),
                "stl_best_val": comp.get("stl_best_val"),
                "winner": comp.get("winner"),
                "winner_phase": comp.get("winner_phase"),
                "winner_value": comp.get("winner_value"),
                "search_architecture": format_architecture_for_report(comp.get("search_architecture")),
                "stl_architecture": format_architecture_for_report(comp.get("stl_architecture")),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="DAE/DNN exhaustive benchmark search suite")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="DAE/DNN/results")
    p.add_argument("--run-root", type=str, default=None)
    p.add_argument("--tasks", type=str, nargs="+", default=["all"])
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["grid", "random", "bayes", "nas", *ADP_METHODS],
        help="Search methods to run",
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
        use_bn=not bool(args.no_bn),
        candidate_budget=int(args.candidate_budget),
        bayes_warmup=int(args.bayes_warmup),
        nas_patience=int(args.nas_patience),
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
                        "search_best_val",
                        "stl_best_val",
                        "winner",
                        "winner_phase",
                        "winner_value",
                        "search_architecture",
                        "stl_architecture",
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
