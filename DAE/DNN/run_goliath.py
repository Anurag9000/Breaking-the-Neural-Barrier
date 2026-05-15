from __future__ import annotations

import argparse
import gc
import copy
import csv
import datetime as _dt
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[2]))

from DAE.DNN.adp_search import expand_depth, expand_width, model_depth, model_width
from DAE.DNN.mlp import MLP
from DAE.DNN.tasks import Task, build_task, refresh_task_loaders, task_names
from DAE.DNN.train_utils import eval_epoch, train_epoch, unpack_batch
from DAE.DNN.train_utils import AdaptiveBatchController
from utils.adp_logging import ContinuousLogger
from utils.adp_plot import plot_best_loss_per_neurons_from_csv, plot_val_loss_from_csv

DEFAULT_MAX_EPOCHS = 99999999999999999999999999999999999999999999999999999999999999999999999
DEFAULT_VRAM_BUDGET_GB = 5.5

PER_TASK_BATCH_SIZES = {
    "prediction": 32768,
    "ranking": 32768,
    "representation": 32768,
    "autoencoding": 32768,
    "generation": 32768,
    "denoising": 32768,
    "anomaly": 32768,
    "clustering": 32768,
    "compression": 32768,
    "multimodal": 32768,
    "selfsupervised": 32768,
    "inverse": 32768,
    "control": 32768,
    "simulation": 32768,
    "misc": 32768,
}

GOLIATH_ADP_PHASES = [
    ("ae_alt_width", "alt_width"),
    ("ae_alt_depth", "alt_depth"),
    ("ae_width_only", "width_only"),
    ("ae_depth_only", "depth_only"),
    ("ae_width_to_depth", "width_to_depth"),
    ("ae_depth_to_width", "depth_to_width"),
]

GOLIATH_PHASE_ORDER = [name for name, _ in GOLIATH_ADP_PHASES]


@dataclass
class RunConfig:
    data_dir: str
    results_dir: str
    run_root: Optional[str]
    tasks: List[str]
    phases: List[str]
    batch_size: int
    num_workers: int
    seed: int
    stl_width: int
    stl_depth: int
    alt_start_width: int
    alt_start_depth: int
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
    demo: bool


@dataclass
class CandidateResult:
    best_val: float
    best_epoch: int
    final_epoch: int
    best_checkpoint: Path
    last_checkpoint: Path
    candidate_dir: Path
    architecture: List[int]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def slug(text: str) -> str:
    out = []
    for ch in text.lower():
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def model_signature(model: MLP) -> Dict[str, Any]:
    return {
        "in_dim": int(model.in_dim),
        "hidden_widths": [int(w) for w in model.hidden_widths],
        "out_dim": int(model.out_dim),
        "use_bn": bool(model.use_bn),
    }


def make_model(in_dim: int, hidden_widths: Sequence[int], out_dim: int, use_bn: bool) -> MLP:
    return MLP(in_dim=int(in_dim), hidden_widths=[int(w) for w in hidden_widths], out_dim=int(out_dim), use_bn=bool(use_bn))


def make_reconstruction_model(task: Task, hidden_widths: Sequence[int], use_bn: bool) -> MLP:
    return make_model(task.in_dim, hidden_widths, task.in_dim, use_bn)


def make_stl_model(task: Task, hidden_widths: Sequence[int], use_bn: bool) -> MLP:
    return make_model(task.in_dim, hidden_widths, task.out_dim, use_bn)


def _repeat_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    shape = (int(batch_size),) + (1,) * (x.ndim - 1)
    return x.repeat(shape)


def _batch_fits_cuda(
    *,
    task: Task,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    budget_bytes: int,
    device,
) -> Tuple[bool, Optional[int], Optional[int]]:
    if not torch.cuda.is_available():
        return True, None, None

    sample_batch = next(iter(task.train_loader))
    x, y, _ = unpack_batch(sample_batch)
    x = x.to(device)
    y = y.to(device) if isinstance(y, torch.Tensor) else y

    x_big = _repeat_batch(x, batch_size)
    if isinstance(y, torch.Tensor):
        y_big = _repeat_batch(y, batch_size) if y.ndim > 0 else y.repeat(int(batch_size))
    else:
        y_big = y

    try:
        torch.cuda.empty_cache()
        gc.collect()
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        out = model(x_big)
        loss = task.loss_fn(out, y_big)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        return int(peak_alloc) <= int(budget_bytes), peak_alloc, peak_reserved
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return False, None, None
        raise


def auto_select_batch_size(
    *,
    task_name: str,
    cfg_data_dir: str,
    num_workers: int,
    seed: int,
    hidden_widths: Sequence[int],
    use_bn: bool,
    device,
    budget_gb: float = DEFAULT_VRAM_BUDGET_GB,
) -> int:
    if not torch.cuda.is_available():
        return 64

    probe_task = build_task(task_name, cfg_data_dir, 1, num_workers, seed)
    probe_model = make_stl_model(probe_task, hidden_widths, use_bn).to(device)
    probe_optimizer = torch.optim.AdamW(probe_model.parameters(), lr=1e-3, weight_decay=1e-4)
    base_model_state = copy.deepcopy(probe_model.state_dict())
    base_optim_state = copy.deepcopy(probe_optimizer.state_dict())
    budget_bytes = int(float(budget_gb) * (1024**3))

    low = 2
    high = 2
    best_alloc = None
    best_reserved = None
    probe_model.load_state_dict(base_model_state)
    probe_optimizer.load_state_dict(base_optim_state)
    fits, best_alloc, best_reserved = _batch_fits_cuda(
        task=probe_task,
        model=probe_model,
        optimizer=probe_optimizer,
        batch_size=2,
        budget_bytes=budget_bytes,
        device=device,
    )
    best = 2 if fits else 2

    while True:
        probe_model.load_state_dict(base_model_state)
        probe_optimizer.load_state_dict(base_optim_state)
        fits, alloc, reserved = _batch_fits_cuda(
            task=probe_task,
            model=probe_model,
            optimizer=probe_optimizer,
            batch_size=high,
            budget_bytes=budget_bytes,
            device=device,
        )
        if not fits:
            break
        best = high
        best_alloc = alloc
        best_reserved = reserved
        low = high
        high *= 2
        if high > 1 << 18:
            break

    while low + 1 < high:
        mid = (low + high) // 2
        probe_model.load_state_dict(base_model_state)
        probe_optimizer.load_state_dict(base_optim_state)
        fits, alloc, reserved = _batch_fits_cuda(
            task=probe_task,
            model=probe_model,
            optimizer=probe_optimizer,
            batch_size=mid,
            budget_bytes=budget_bytes,
            device=device,
        )
        if fits:
            best = mid
            best_alloc = alloc
            best_reserved = reserved
            low = mid
        else:
            high = mid

    auto_selected = max(2, best // 2)
    print(
        f"Batch probe for {task_name}: best_safe={best}, half_default={auto_selected}, "
        f"peak_allocated={best_alloc if best_alloc is not None else 'na'} bytes, "
        f"peak_reserved={best_reserved if best_reserved is not None else 'na'} bytes"
    )
    return auto_selected


def base_stl_hidden(task: Task, cfg: RunConfig) -> List[int]:
    width = int(cfg.stl_width)
    if "max_width" in task.extra:
        width = min(width, int(task.extra["max_width"]))
    width = max(2, width)
    depth = max(1, int(cfg.stl_depth))
    return [width for _ in range(depth)]


def adp_seed_hidden() -> List[int]:
    return [2, 2]


def default_batch_size_for_task(task_name: str) -> int:
    return int(PER_TASK_BATCH_SIZES.get(task_name.lower(), 32768))


def batch_size_for_task(task_name: str, override: int) -> int:
    override = int(override)
    if override > 0:
        return override
    return default_batch_size_for_task(task_name)


def candidate_slug(candidate_index: int, hidden_widths: Sequence[int]) -> str:
    depth = len(hidden_widths)
    width = max(hidden_widths) if hidden_widths else 0
    return f"cand_{candidate_index:03d}_d{depth}_w{width}"


def phase_root_for(task_root: Path, phase_name: str) -> Path:
    return task_root / phase_name


def candidate_root_for(phase_root: Path, candidate_index: int, hidden_widths: Sequence[int]) -> Path:
    return phase_root / candidate_slug(candidate_index, hidden_widths)


def reconstruction_train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn,
    optimizer,
    device,
    grad_clip: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        x, _, _ = unpack_batch(batch)
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, x)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return float(total_loss / max(total, 1))


@torch.no_grad()
def reconstruction_eval_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn,
    device,
    measure_throughput: bool = False,
):
    import time

    model.eval()
    total_loss = 0.0
    total = 0
    start = time.time()
    for batch in loader:
        x, _, _ = unpack_batch(batch)
        x = x.to(device)
        out = model(x)
        loss = loss_fn(out, x)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    end = time.time()
    throughput = None
    if measure_throughput and total > 0:
        throughput = float(total / max(end - start, 1e-6))
    return float(total_loss / max(total, 1)), None, throughput


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    best_state: Dict[str, torch.Tensor],
    best_epoch: int,
    es_counter: int,
    metadata: Dict[str, Any],
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "best_val": float(best_val),
        "best_state": copy.deepcopy(best_state),
        "best_epoch": int(best_epoch),
        "es_counter": int(es_counter),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def infer_model_signature_from_state_dict(state_dict: Dict[str, Any]) -> Tuple[int, List[int], int, bool]:
    linear_shapes: List[Tuple[int, int]] = []
    use_bn = False
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.ndim == 2:
            linear_shapes.append((int(value.shape[0]), int(value.shape[1])))
        if "running_mean" in key or "running_var" in key:
            use_bn = True
    if not linear_shapes:
        raise ValueError("Could not infer MLP architecture from checkpoint state dict")
    in_dim = int(linear_shapes[0][1])
    hidden_widths = [int(shape[0]) for shape in linear_shapes[:-1]]
    out_dim = int(linear_shapes[-1][0])
    return in_dim, hidden_widths, out_dim, use_bn


def phase_metadata(
    *,
    task: Task,
    phase_name: str,
    phase_kind: str,
    reconstruct: bool,
    model: MLP,
    cfg: RunConfig,
    candidate_index: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "task": task.name,
        "phase_name": phase_name,
        "phase_kind": phase_kind,
        "reconstruct": reconstruct,
        "candidate_index": int(candidate_index),
        "model": model_signature(model),
        "task_type": task.task_type,
        "task_extra": task.extra,
        "config": asdict(cfg),
        "git_commit": git_commit(),
        "dataset_sizes": {
            "train": len(task.train_loader.dataset) if hasattr(task.train_loader, "dataset") else None,
            "val": len(task.val_loader.dataset) if hasattr(task.val_loader, "dataset") else None,
            "test": len(task.test_loader.dataset) if hasattr(task.test_loader, "dataset") else None,
        },
    }
    if extra:
        payload.update(extra)
    return payload


def checkpoint_resume_ready(candidate_dir: Path) -> bool:
    state_path = candidate_dir / "candidate_state.json"
    ckpt = candidate_dir / "checkpoint_last.pt"
    return state_path.exists() and ckpt.exists() and read_json(state_path).get("completed", False) is False


def candidate_completed(candidate_dir: Path) -> bool:
    state_path = candidate_dir / "candidate_state.json"
    return state_path.exists() and read_json(state_path).get("completed", False) is True


def load_candidate_model(candidate_dir: Path, device) -> Tuple[MLP, Dict[str, Any], Dict[str, Any]]:
    meta_path = candidate_dir / "metadata.json"
    ckpt = load_checkpoint(candidate_dir / "checkpoint_best.pt", device)
    state_dict = ckpt.get("best_state") or ckpt.get("model_state")
    if state_dict is None:
        raise ValueError(f"Checkpoint at {candidate_dir} does not contain model weights")

    if meta_path.exists():
        meta = read_json(meta_path)
        model = make_model(
            meta["model"]["in_dim"],
            meta["model"]["hidden_widths"],
            meta["model"]["out_dim"],
            meta["model"]["use_bn"],
        ).to(device)
    else:
        in_dim, hidden_widths, out_dim, use_bn = infer_model_signature_from_state_dict(state_dict)
        model = make_model(in_dim, hidden_widths, out_dim, use_bn).to(device)
        meta = {
            "candidate_dir": str(candidate_dir),
            "model": {
                "in_dim": int(in_dim),
                "hidden_widths": [int(w) for w in hidden_widths],
                "out_dim": int(out_dim),
                "use_bn": bool(use_bn),
            },
            "source": "inferred_from_checkpoint",
        }

    model.load_state_dict(state_dict)
    return model, meta, ckpt


def resolve_candidate_dir(phase_root: Path, candidate_ref: Optional[str]) -> Optional[Path]:
    if not candidate_ref:
        return None
    path = Path(candidate_ref)
    if path.exists():
        return path
    return phase_root / candidate_ref


def training_loop(
    *,
    task: Task,
    model: MLP,
    candidate_dir: Path,
    cfg: RunConfig,
    device,
    logger: ContinuousLogger,
    reconstruct: bool,
    resume: bool = True,
    batch_controller: Optional[AdaptiveBatchController] = None,
) -> CandidateResult:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = candidate_dir / "metadata.json"
    candidate_state_path = candidate_dir / "candidate_state.json"
    last_ckpt = candidate_dir / "checkpoint_last.pt"
    best_ckpt = candidate_dir / "checkpoint_best.pt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    start_epoch = 1
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    es_counter = 0
    metric_keys: List[str] = []
    current_batch_size = int(getattr(task.train_loader, "batch_size", 0) or 0)
    if batch_controller is not None:
        refreshed = int(batch_controller.current_batch_size)
        if refreshed > 0 and refreshed != current_batch_size:
            refresh_task_loaders(task, refreshed)
            current_batch_size = refreshed

    if resume and last_ckpt.exists():
        ckpt = load_checkpoint(last_ckpt, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["best_val"])
        best_state = copy.deepcopy(ckpt["best_state"])
        best_epoch = int(ckpt["best_epoch"])
        es_counter = int(ckpt["es_counter"])

    if not metadata_path.exists():
        write_json(metadata_path, phase_metadata(task=task, phase_name=candidate_dir.parent.name, phase_kind="candidate", reconstruct=reconstruct, model=model, cfg=cfg, candidate_index=int(candidate_dir.name.split("_")[1])))

    if start_epoch > int(cfg.max_epochs):
        model.load_state_dict(best_state)
        write_json(
            candidate_state_path,
            {
                "completed": True,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "final_epoch": start_epoch - 1,
                "candidate_dir": str(candidate_dir),
                "checkpoint_best": str(best_ckpt),
                "checkpoint_last": str(last_ckpt),
                "architecture": [int(w) for w in model.hidden_widths],
                "reconstruct": reconstruct,
            },
        )
        return CandidateResult(best_val, best_epoch, start_epoch - 1, best_ckpt, last_ckpt, candidate_dir, [int(w) for w in model.hidden_widths])

    for epoch in range(start_epoch, int(cfg.max_epochs) + 1):
        if batch_controller is not None:
            batch_controller.maybe_poll()
            refreshed = int(batch_controller.current_batch_size)
            if refreshed > 0 and refreshed != current_batch_size:
                refresh_task_loaders(task, refreshed)
                current_batch_size = refreshed

        if reconstruct:
            tr_loss = reconstruction_train_epoch(model, task.train_loader, F.mse_loss, optimizer, device, grad_clip=float(cfg.grad_clip))
            val_loss, val_acc, throughput = reconstruction_eval_epoch(
                model, task.val_loader, F.mse_loss, device, measure_throughput=False
            )
            tr_acc = None
        else:
            tr_loss, tr_acc = train_epoch(model, task.train_loader, task.loss_fn, optimizer, device, task.task_type, float(cfg.grad_clip))
            val_loss, val_acc, throughput = eval_epoch(model, task.val_loader, task.loss_fn, device, task.task_type, measure_throughput=False)

        metrics: Dict[str, Any] = {}
        if not reconstruct and task.metrics_fn is not None and (epoch == 1 or epoch % 5 == 0):
            metrics = task.metrics_fn(model, task, device) or {}
            if not metric_keys:
                metric_keys = list(metrics.keys())

        improved = val_loss < (best_val - float(cfg.delta))
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            es_counter = 0
            save_checkpoint(
                best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val=best_val,
                best_state=best_state,
                best_epoch=best_epoch,
                es_counter=es_counter,
                metadata={
                    "task": task.name,
                    "phase": candidate_dir.parent.name,
                    "candidate_dir": str(candidate_dir),
                    "reconstruct": reconstruct,
                },
            )
        else:
            es_counter += 1

        save_checkpoint(
            last_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val=best_val,
            best_state=best_state,
            best_epoch=best_epoch,
            es_counter=es_counter,
            metadata={
                "task": task.name,
                "phase": candidate_dir.parent.name,
                "candidate_dir": str(candidate_dir),
                "reconstruct": reconstruct,
            },
        )

        row: Dict[str, Any] = {
            "task": task.name,
            "phase": candidate_dir.parent.name,
            "candidate_dir": candidate_dir.name,
            "epoch": epoch,
            "width": model_width(model),
            "depth": model_depth(model),
            "neurons": int(sum(model.hidden_widths) + model.out_dim),
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "es_counter": es_counter,
            "improved": improved,
            "train_acc": tr_acc,
            "val_acc": val_acc,
            "throughput": throughput,
        }
        for key in metric_keys:
            row[key] = metrics.get(key)
        logger.log_epoch_stats(row)
        logger.log_console(
            f"[{task.name}][{candidate_dir.parent.name}][{candidate_dir.name}] epoch={epoch} train_loss={tr_loss:.6f} val_loss={val_loss:.6f} best={best_val:.6f} es={es_counter}/{cfg.patience}"
        )

        write_json(
            candidate_state_path,
            {
                "completed": False,
                "task": task.name,
                "phase": candidate_dir.parent.name,
                "candidate_dir": str(candidate_dir),
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "final_epoch": epoch,
                "es_counter": es_counter,
                "architecture": [int(w) for w in model.hidden_widths],
                "reconstruct": reconstruct,
                "checkpoint_best": str(best_ckpt),
                "checkpoint_last": str(last_ckpt),
            },
        )

        if es_counter >= int(cfg.patience):
            break

    model.load_state_dict(best_state)
    save_checkpoint(
        best_ckpt,
        model=model,
        optimizer=optimizer,
        epoch=best_epoch,
        best_val=best_val,
        best_state=best_state,
        best_epoch=best_epoch,
        es_counter=0,
        metadata={"task": task.name, "phase": candidate_dir.parent.name, "candidate_dir": str(candidate_dir), "reconstruct": reconstruct},
    )
    write_json(
        candidate_state_path,
        {
            "completed": True,
            "task": task.name,
            "phase": candidate_dir.parent.name,
            "candidate_dir": str(candidate_dir),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "final_epoch": epoch,
            "architecture": [int(w) for w in model.hidden_widths],
            "reconstruct": reconstruct,
            "checkpoint_best": str(best_ckpt),
            "checkpoint_last": str(last_ckpt),
        },
    )

    return CandidateResult(best_val, best_epoch, epoch, best_ckpt, last_ckpt, candidate_dir, [int(w) for w in model.hidden_widths])


def phase_progress_header() -> List[str]:
    return [
        "task",
        "phase",
        "candidate_index",
        "candidate_dir",
        "architecture",
        "best_val",
        "best_epoch",
        "final_epoch",
        "best_checkpoint",
        "last_checkpoint",
        "improved_over_global",
        "search_phase",
        "width_fail",
        "depth_fail",
    ]


def log_phase_progress(path: Path, row: Dict[str, Any]) -> None:
    append_csv_row(path, row)


def plot_candidate_stats(candidate_dir: Path, title_prefix: str) -> None:
    csv_path = candidate_dir / "training_stats.csv"
    if not csv_path.exists():
        print(f"Skipping plots for {candidate_dir}: missing {csv_path}")
        return
    try:
        plot_val_loss_from_csv(csv_path, candidate_dir / "val_loss_vs_step.png", title=f"{title_prefix} val_loss")
        plot_best_loss_per_neurons_from_csv(
            csv_path, candidate_dir / "loss_vs_neurons_best.png", title=f"{title_prefix} best val_loss per neurons"
        )
    except FileNotFoundError:
        print(f"Skipping plots for {candidate_dir}: could not read {csv_path}")


def eval_final(model: MLP, task: Task, device, reconstruct: bool) -> Dict[str, Any]:
    if reconstruct:
        val_loss, _, _ = reconstruction_eval_epoch(model, task.test_loader, F.mse_loss, device, measure_throughput=False)
        return {"test_loss": val_loss}
    val_loss, val_acc, _ = eval_epoch(model, task.test_loader, task.loss_fn, device, task.task_type, measure_throughput=False)
    out = {"test_loss": val_loss}
    if val_acc is not None:
        out["test_acc"] = val_acc
    if task.metrics_fn is not None:
        metrics = task.metrics_fn(model, task, device) or {}
        out.update(metrics)
    return out


def latest_completed_candidate(phase_root: Path) -> Optional[Path]:
    candidate_dirs = sorted([p for p in phase_root.iterdir() if p.is_dir() and p.name.startswith("cand_")])
    for cand in reversed(candidate_dirs):
        state_path = cand / "candidate_state.json"
        if state_path.exists() and read_json(state_path).get("completed", False):
            return cand
    return None


def incomplete_candidate(phase_root: Path) -> Optional[Path]:
    candidate_dirs = sorted([p for p in phase_root.iterdir() if p.is_dir() and p.name.startswith("cand_")])
    for cand in reversed(candidate_dirs):
        state_path = cand / "candidate_state.json"
        if state_path.exists() and not read_json(state_path).get("completed", False):
            return cand
    return None


def ensure_phase_state(phase_root: Path, mode: str) -> Dict[str, Any]:
    state_path = phase_root / "search_state.json"
    if state_path.exists():
        return read_json(state_path)
    initial_phase = "width" if mode in ["width_only", "alt_width", "width_to_depth"] else "depth"
    state = {
        "mode": mode,
        "current_phase": initial_phase,
        "width_fail": 0,
        "depth_fail": 0,
        "best_val": 1e30,
        "candidate_index": 0,
        "completed": False,
        "best_candidate_dir": None,
        "best_checkpoint": None,
    }
    write_json(state_path, state)
    return state


def save_phase_state(phase_root: Path, state: Dict[str, Any]) -> None:
    write_json(phase_root / "search_state.json", state)


def can_widen(model: MLP, cfg: RunConfig) -> bool:
    if not model.hidden_widths:
        return False
    return model_width(model) + 1 <= int(cfg.max_width) and int(sum(model.hidden_widths) + model.out_dim) < int(cfg.max_neurons)


def can_deepen(model: MLP, cfg: RunConfig) -> bool:
    if not model.hidden_widths:
        return False
    return len(model.hidden_widths) + 1 <= int(cfg.max_depth) and int(sum(model.hidden_widths) + model.hidden_widths[-1] + model.out_dim) <= int(cfg.max_neurons)


def infer_hidden_from_checkpoint(candidate_dir: Path) -> List[int]:
    meta_path = candidate_dir / "metadata.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        return [int(w) for w in meta["model"]["hidden_widths"]]
    ckpt = load_checkpoint(candidate_dir / "checkpoint_best.pt", device="cpu")
    state_dict = ckpt.get("best_state") or ckpt.get("model_state")
    if state_dict is None:
        raise ValueError(f"Checkpoint at {candidate_dir} does not contain model weights")
    _, hidden_widths, _, _ = infer_model_signature_from_state_dict(state_dict)
    return hidden_widths


def phase_mode(phase_name: str) -> Optional[str]:
    if phase_name == "stl":
        return None
    for name, mode in GOLIATH_ADP_PHASES:
        if phase_name == name:
            return mode
    raise ValueError(f"Unknown phase: {phase_name}")


def phase_seed_hidden(phase_name: str, task: Task, cfg: RunConfig) -> List[int]:
    if phase_name == "stl":
        return base_stl_hidden(task, cfg)
    return adp_seed_hidden()


def extract_hidden_widths(architecture: Any) -> List[int]:
    if isinstance(architecture, dict):
        if "hidden_widths" in architecture:
            return [int(w) for w in architecture["hidden_widths"]]
    if isinstance(architecture, (list, tuple)):
        return [int(w) for w in architecture]
    raise ValueError(f"Unsupported architecture payload: {architecture!r}")


def format_architecture_for_report(architecture: Any) -> str:
    if architecture is None:
        return "n/a"
    if isinstance(architecture, dict):
        hidden_widths = architecture.get("hidden_widths")
        if hidden_widths is not None:
            in_dim = architecture.get("in_dim", "?")
            out_dim = architecture.get("out_dim", "?")
            use_bn = architecture.get("use_bn", "?")
            return f"in={in_dim} hidden={list(hidden_widths)} out={out_dim} bn={use_bn}"
        return json.dumps(architecture, sort_keys=True)
    if isinstance(architecture, (list, tuple)):
        return str([int(w) for w in architecture])
    return str(architecture)


def format_float_for_report(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def build_task_report(task_summary: Dict[str, Any]) -> Dict[str, Any]:
    adp_runs = {entry.get("phase"): entry for entry in task_summary.get("adp_runs", [])}
    paired_stl_runs = {entry.get("phase"): entry for entry in task_summary.get("paired_stl_runs", [])}
    baseline_runs = task_summary.get("baseline_stl_runs", [])
    baseline_comparisons = [comp for comp in task_summary.get("comparisons", []) if comp.get("adp_phase") is None]

    adp_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    for adp_phase_name, adp_mode in GOLIATH_ADP_PHASES:
        adp_entry = adp_runs.get(adp_phase_name)
        if adp_entry is None:
            continue
        stl_entry = paired_stl_runs.get(f"stl_from_{adp_phase_name}")
        comparison = next(
            (comp for comp in task_summary.get("comparisons", []) if comp.get("adp_phase") == adp_phase_name),
            None,
        )
        adp_rows.append(
            {
                "phase": adp_phase_name,
                "mode": adp_mode,
                "architecture": adp_entry.get("architecture"),
                "best_val": adp_entry.get("best_val"),
                "best_epoch": adp_entry.get("best_epoch"),
                "final_epoch": adp_entry.get("final_epoch"),
                "test_metrics": adp_entry.get("test_metrics"),
                "candidate_dir": adp_entry.get("best_candidate_dir"),
            }
        )
        paired_rows.append(
            {
                "adp_phase": adp_phase_name,
                "stl_phase": f"stl_from_{adp_phase_name}",
                "adp_best_val": None if comparison is None else comparison.get("adp_best_val"),
                "stl_best_val": None if comparison is None else comparison.get("stl_best_val"),
                "winner": None if comparison is None else comparison.get("winner"),
                "winner_phase": None if comparison is None else comparison.get("winner_phase"),
                "winner_value": None if comparison is None else comparison.get("winner_value"),
                "adp_architecture": None if comparison is None else comparison.get("adp_architecture"),
                "stl_architecture": None if comparison is None else comparison.get("stl_architecture"),
                "stl_run": stl_entry,
            }
        )

    return {
        "task": task_summary.get("task"),
        "adp_runs": adp_rows,
        "paired_stl_runs": paired_rows,
        "baseline_stl_runs": baseline_runs,
        "baseline_comparisons": baseline_comparisons,
        "comparisons": task_summary.get("comparisons", []),
        "winner": task_summary.get("winner"),
    }


def render_final_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# DAE/DNN Goliath Final Report")
    lines.append("")
    lines.append(f"- Run root: `{report.get('run_root', 'n/a')}`")
    lines.append(f"- Git commit: `{report.get('git_commit', 'n/a')}`")
    lines.append(f"- Device: `{report.get('device', 'n/a')}`")
    lines.append(f"- Tasks completed: `{report.get('summary', {}).get('tasks_completed', 0)}`")
    lines.append("")

    for task_report in report.get("tasks", []):
        lines.append(f"## Task: {task_report.get('task', 'n/a')}")
        winner = task_report.get("winner") or {}
        lines.append(
            f"- Overall winner: `{winner.get('winner', 'n/a')}` via `{winner.get('winner_phase', 'n/a')}` "
            f"at `{format_float_for_report(winner.get('winner_value'))}`"
        )
        if winner.get("adp_architecture") is not None:
            lines.append(f"- Winner ADP architecture: `{format_architecture_for_report(winner.get('adp_architecture'))}`")
        if winner.get("stl_architecture") is not None:
            lines.append(f"- Winner STL architecture: `{format_architecture_for_report(winner.get('stl_architecture'))}`")
        lines.append("")
        lines.append("| ADP variant | ADP best arch | ADP best val | STL refit arch | STL refit best val | Winner |")
        lines.append("|---|---|---:|---|---:|---|")
        for adp_entry, paired_entry in zip(task_report.get("adp_runs", []), task_report.get("paired_stl_runs", [])):
            winner_label = paired_entry.get("winner") or "n/a"
            lines.append(
                "| "
                f"{adp_entry.get('phase', 'n/a')} | "
                f"{format_architecture_for_report(adp_entry.get('architecture'))} | "
                f"{format_float_for_report(adp_entry.get('best_val'))} | "
                f"{format_architecture_for_report(paired_entry.get('stl_architecture'))} | "
                f"{format_float_for_report(paired_entry.get('stl_best_val'))} | "
                f"{winner_label} |"
            )
        if task_report.get("baseline_stl_runs"):
            lines.append("")
            lines.append("Standalone STL baseline runs:")
            for baseline in task_report.get("baseline_stl_runs", []):
                lines.append(
                    f"- `{baseline.get('phase', 'stl_base')}` arch `{format_architecture_for_report(baseline.get('architecture'))}` "
                    f"best val `{format_float_for_report(baseline.get('best_val'))}`"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_task_summary_csv_rows(task_name: str, task_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    comparisons = {entry.get("adp_phase"): entry for entry in task_summary.get("comparisons", []) if entry.get("adp_phase") is not None}
    for adp_phase_name, adp_mode in GOLIATH_ADP_PHASES:
        comp = comparisons.get(adp_phase_name)
        if comp is None:
            continue
        rows.append(
            {
                "task": task_name,
                "row_type": "adp_vs_stl",
                "adp_phase": adp_phase_name,
                "adp_mode": adp_mode,
                "stl_phase": comp.get("stl_phase"),
                "adp_best_val": comp.get("adp_best_val"),
                "stl_best_val": comp.get("stl_best_val"),
                "winner": comp.get("winner"),
                "winner_phase": comp.get("winner_phase"),
                "winner_value": comp.get("winner_value"),
                "adp_architecture": format_architecture_for_report(comp.get("adp_architecture")),
                "stl_architecture": format_architecture_for_report(comp.get("stl_architecture")),
            }
        )

    for baseline in task_summary.get("baseline_stl_runs", []):
        rows.append(
            {
                "task": task_name,
                "row_type": "baseline_stl",
                "adp_phase": "",
                "adp_mode": "",
                "stl_phase": baseline.get("phase"),
                "adp_best_val": "",
                "stl_best_val": baseline.get("best_val"),
                "winner": "stl",
                "winner_phase": baseline.get("phase"),
                "winner_value": baseline.get("best_val"),
                "adp_architecture": "",
                "stl_architecture": format_architecture_for_report(baseline.get("architecture")),
            }
        )
    return rows


def run_stl_phase(
    task: Task,
    task_root: Path,
    cfg: RunConfig,
    device,
    base_hidden: List[int],
    phase_name: str = "stl",
    source_phase: Optional[str] = None,
    batch_controller: Optional[AdaptiveBatchController] = None,
) -> Dict[str, Any]:
    phase_root = phase_root_for(task_root, phase_name)
    phase_root.mkdir(parents=True, exist_ok=True)
    progress_path = phase_root / "phase_progress.csv"
    state = ensure_phase_state(phase_root, phase_name)
    summary_path = phase_root / "phase_summary.json"

    if state.get("completed", False) and summary_path.exists():
        return read_json(summary_path)

    candidate_idx = int(state.get("candidate_index", 0))
    candidate_dir = candidate_root_for(phase_root, candidate_idx, base_hidden)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    model = make_stl_model(task, base_hidden, cfg.use_bn).to(device)
    logger = ContinuousLogger(candidate_dir, f"{task.name}_{phase_name}", phase_name)
    write_json(
        phase_root / "phase_metadata.json",
        phase_metadata(
            task=task,
            phase_name=phase_name,
            phase_kind="stl",
            reconstruct=False,
            model=model,
            cfg=cfg,
            candidate_index=candidate_idx,
            extra={"hidden_widths": base_hidden, "seed_hidden_widths": base_hidden, "source_phase": source_phase},
        ),
    )

    if candidate_completed(candidate_dir):
        model, meta, ckpt = load_candidate_model(candidate_dir, device)
        test_metrics = eval_final(model, task, device, reconstruct=False)
        summary = {
            "task": task.name,
            "phase": phase_name,
            "source_phase": source_phase,
            "candidate_dir": candidate_dir.name,
            "architecture": base_hidden,
            "best_val": float(ckpt["best_val"]),
            "best_epoch": int(ckpt["best_epoch"]),
            "test_metrics": test_metrics,
            "checkpoint_best": str(candidate_dir / "checkpoint_best.pt"),
            "checkpoint_last": str(candidate_dir / "checkpoint_last.pt"),
            "resumed": True,
        }
        write_json(summary_path, summary)
        log_phase_progress(
            progress_path,
            {
                "task": task.name,
                "phase": phase_name,
                "candidate_index": candidate_idx,
                "candidate_dir": candidate_dir.name,
                "architecture": str(base_hidden),
                "best_val": float(ckpt["best_val"]),
                "best_epoch": int(ckpt["best_epoch"]),
                "final_epoch": int(ckpt["epoch"]) if "epoch" in ckpt else int(ckpt["best_epoch"]),
                "best_checkpoint": str(candidate_dir / "checkpoint_best.pt"),
                "last_checkpoint": str(candidate_dir / "checkpoint_last.pt"),
                "improved_over_global": True,
                "search_phase": phase_name,
                "width_fail": 0,
                "depth_fail": 0,
            },
        )
        logger.close()
        return summary

    result = training_loop(
        task=task,
        model=model,
        candidate_dir=candidate_dir,
        cfg=cfg,
        device=device,
        logger=logger,
        reconstruct=False,
        resume=True,
        batch_controller=batch_controller,
    )
    logger.close()
    test_metrics = eval_final(model, task, device, reconstruct=False)
    summary = {
        "task": task.name,
        "phase": phase_name,
        "source_phase": source_phase,
        "candidate_dir": candidate_dir.name,
        "architecture": base_hidden,
        "best_val": float(result.best_val),
        "best_epoch": int(result.best_epoch),
        "final_epoch": int(result.final_epoch),
        "test_metrics": test_metrics,
        "checkpoint_best": str(result.best_checkpoint),
        "checkpoint_last": str(result.last_checkpoint),
        "resumed": False,
    }
    write_json(summary_path, summary)
    log_phase_progress(
        progress_path,
        {
            "task": task.name,
            "phase": phase_name,
            "candidate_index": candidate_idx,
            "candidate_dir": candidate_dir.name,
            "architecture": str(base_hidden),
            "best_val": float(result.best_val),
            "best_epoch": int(result.best_epoch),
            "final_epoch": int(result.final_epoch),
            "best_checkpoint": str(result.best_checkpoint),
            "last_checkpoint": str(result.last_checkpoint),
            "improved_over_global": True,
            "search_phase": phase_name,
            "width_fail": 0,
            "depth_fail": 0,
        },
    )
    plot_candidate_stats(candidate_dir, f"{task.name} {phase_name}")
    state.update({"candidate_index": candidate_idx, "completed": True, "best_val": float(result.best_val), "best_candidate_dir": candidate_dir.name, "best_checkpoint": str(result.best_checkpoint)})
    save_phase_state(phase_root, state)
    return summary


def run_growth_phase(task: Task, task_root: Path, cfg: RunConfig, device, base_hidden: List[int], phase_name: str, mode: str, reconstruct: bool, batch_controller: Optional[AdaptiveBatchController] = None) -> Dict[str, Any]:
    phase_root = phase_root_for(task_root, phase_name)
    phase_root.mkdir(parents=True, exist_ok=True)
    progress_path = phase_root / "phase_progress.csv"
    summary_path = phase_root / "phase_summary.json"
    state_file = phase_root / "search_state.json"
    had_state = state_file.exists()
    state = ensure_phase_state(phase_root, mode)

    if state.get("completed", False) and summary_path.exists():
        return read_json(summary_path)

    candidate_dirs = sorted([p for p in phase_root.iterdir() if p.is_dir() and p.name.startswith("cand_")])
    if not had_state and candidate_dirs:
        next_candidate_index = max(int(p.name.split("_")[1]) for p in candidate_dirs) + 1
    else:
        next_candidate_index = int(state.get("candidate_index", 0))
    current_phase = state.get("current_phase", "width" if mode in ["width_only", "alt_width", "width_to_depth"] else "depth")
    global_best_val = float(state.get("best_val", 1e30))
    width_fail = int(state.get("width_fail", 0))
    depth_fail = int(state.get("depth_fail", 0))
    global_best_candidate_dir = resolve_candidate_dir(phase_root, state.get("best_candidate_dir"))
    global_best_checkpoint = Path(state["best_checkpoint"]) if state.get("best_checkpoint") else None

    if global_best_candidate_dir is None:
        completed_dirs = [p for p in candidate_dirs if candidate_completed(p)]
        if completed_dirs:
            scored: List[Tuple[float, Path]] = []
            for cand in completed_dirs:
                cand_state = read_json(cand / "candidate_state.json")
                scored.append((float(cand_state.get("best_val", 1e30)), cand))
            scored.sort(key=lambda item: item[0])
            global_best_val, global_best_candidate_dir = scored[0]
            global_best_checkpoint = global_best_candidate_dir / "checkpoint_best.pt"

    def current_base_model() -> MLP:
        latest = latest_completed_candidate(phase_root)
        if latest is not None:
            base, _, _ = load_candidate_model(latest, device)
            return base
        return make_reconstruction_model(task, base_hidden, cfg.use_bn).to(device) if reconstruct else make_stl_model(task, base_hidden, cfg.use_bn).to(device)

    # Resume an incomplete candidate first.
    incomplete = incomplete_candidate(phase_root)
    if incomplete is not None:
        candidate_idx = int(incomplete.name.split("_")[1])
        candidate_model, meta, ckpt = load_candidate_model(incomplete, device)
        logger = ContinuousLogger(incomplete, f"{task.name}_{phase_name}", phase_name)
        result = training_loop(
            task=task,
            model=candidate_model,
            candidate_dir=incomplete,
            cfg=cfg,
            device=device,
            logger=logger,
            reconstruct=reconstruct,
            resume=True,
            batch_controller=batch_controller,
        )
        logger.close()
        state["candidate_index"] = candidate_idx + 1
        if result.best_val < (global_best_val - float(cfg.delta)):
            global_best_val = float(result.best_val)
            global_best_candidate_dir = incomplete
            global_best_checkpoint = incomplete / "checkpoint_best.pt"
        state["best_candidate_dir"] = global_best_candidate_dir.name if global_best_candidate_dir is not None else None
        state["best_checkpoint"] = str(global_best_checkpoint) if global_best_checkpoint is not None else None
        state["best_val"] = global_best_val
        state["current_phase"] = current_phase
        save_phase_state(phase_root, state)
        candidate_dirs = sorted([p for p in phase_root.iterdir() if p.is_dir() and p.name.startswith("cand_")])

    # Recompute after any resumed candidate.
    while True:
        completed_dirs = [p for p in candidate_dirs if candidate_completed(p)]
        if completed_dirs:
            latest = completed_dirs[-1]
            latest_model, latest_meta, latest_ckpt = load_candidate_model(latest, device)
            current_base = latest_model
        else:
            current_base = current_base_model()

        if next_candidate_index == 0:
            next_model = current_base
            next_arch = [int(w) for w in next_model.hidden_widths]
        else:
            if mode == "width_only":
                if not can_widen(current_base, cfg) or width_fail >= int(cfg.patience):
                    break
                next_model = expand_width(current_base, 1, int(cfg.max_width))
                if next_model is None:
                    break
                next_arch = [int(w) for w in next_model.hidden_widths]
            elif mode == "depth_only":
                if not can_deepen(current_base, cfg) or depth_fail >= int(cfg.patience):
                    break
                next_model = expand_depth(current_base, int(cfg.max_depth))
                if next_model is None:
                    break
                next_arch = [int(w) for w in next_model.hidden_widths]
            elif mode == "alt_width":
                if current_phase == "width":
                    if not can_widen(current_base, cfg):
                        width_fail = int(cfg.patience)
                        state.update({"width_fail": width_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "depth"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_model = expand_width(current_base, 1, int(cfg.max_width))
                    if next_model is None:
                        width_fail = int(cfg.patience)
                        state.update({"width_fail": width_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "depth"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
                else:
                    if not can_deepen(current_base, cfg):
                        depth_fail = int(cfg.patience)
                        state.update({"depth_fail": depth_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "width"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_model = expand_depth(current_base, int(cfg.max_depth))
                    if next_model is None:
                        depth_fail = int(cfg.patience)
                        state.update({"depth_fail": depth_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "width"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
            elif mode == "alt_depth":
                if current_phase == "depth":
                    if not can_deepen(current_base, cfg):
                        depth_fail = int(cfg.patience)
                        state.update({"depth_fail": depth_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "width"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_model = expand_depth(current_base, int(cfg.max_depth))
                    if next_model is None:
                        depth_fail = int(cfg.patience)
                        state.update({"depth_fail": depth_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "width"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
                else:
                    if not can_widen(current_base, cfg):
                        width_fail = int(cfg.patience)
                        state.update({"width_fail": width_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "depth"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_model = expand_width(current_base, 1, int(cfg.max_width))
                    if next_model is None:
                        width_fail = int(cfg.patience)
                        state.update({"width_fail": width_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "depth"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
            elif mode == "width_to_depth":
                if current_phase == "width":
                    if not can_widen(current_base, cfg):
                        width_fail = int(cfg.patience)
                        state.update({"width_fail": width_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "depth"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_model = expand_width(current_base, 1, int(cfg.max_width))
                    if next_model is None:
                        width_fail = int(cfg.patience)
                        state.update({"width_fail": width_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "depth"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
                else:
                    if not can_deepen(current_base, cfg):
                        depth_fail = int(cfg.patience)
                        state.update({"depth_fail": depth_fail})
                        save_phase_state(phase_root, state)
                        continue
                    next_model = expand_depth(current_base, int(cfg.max_depth))
                    if next_model is None:
                        depth_fail = int(cfg.patience)
                        state.update({"depth_fail": depth_fail})
                        save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
            elif mode == "depth_to_width":
                if current_phase == "depth":
                    if not can_deepen(current_base, cfg):
                        depth_fail = int(cfg.patience)
                        state.update({"depth_fail": depth_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "width"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_model = expand_depth(current_base, int(cfg.max_depth))
                    if next_model is None:
                        depth_fail = int(cfg.patience)
                        state.update({"depth_fail": depth_fail})
                        save_phase_state(phase_root, state)
                        current_phase = "width"
                        state["current_phase"] = current_phase
                        save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
                else:
                    if not can_widen(current_base, cfg):
                        width_fail = int(cfg.patience)
                        state.update({"width_fail": width_fail})
                        save_phase_state(phase_root, state)
                        continue
                    next_model = expand_width(current_base, 1, int(cfg.max_width))
                    if next_model is None:
                        width_fail = int(cfg.patience)
                        state.update({"width_fail": width_fail})
                        save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
            else:
                raise ValueError(f"Unknown mode: {mode}")

        candidate_idx = next_candidate_index
        candidate_dir = candidate_root_for(phase_root, candidate_idx, next_arch)
        logger = ContinuousLogger(candidate_dir, f"{task.name}_{phase_name}", phase_name)
        write_json(
            candidate_dir / "metadata.json",
            phase_metadata(
                task=task,
                phase_name=phase_name,
                phase_kind=mode,
                reconstruct=reconstruct,
                model=next_model,
                cfg=cfg,
                candidate_index=candidate_idx,
                extra={"hidden_widths": next_arch, "search_phase": current_phase},
            ),
        )
        result = training_loop(
            task=task,
            model=next_model.to(device),
            candidate_dir=candidate_dir,
            cfg=cfg,
            device=device,
            logger=logger,
            reconstruct=reconstruct,
            resume=True,
            batch_controller=batch_controller,
        )
        logger.close()

        improved = result.best_val < (global_best_val - float(cfg.delta))
        if improved:
            global_best_val = float(result.best_val)
            global_best_candidate_dir = candidate_dir
            global_best_checkpoint = candidate_dir / "checkpoint_best.pt"
            if mode == "width_only":
                width_fail = 0
            elif mode == "depth_only":
                depth_fail = 0
            else:
                if current_phase == "width":
                    width_fail = 0
                else:
                    depth_fail = 0
        else:
            if mode == "width_only":
                width_fail += 1
            elif mode == "depth_only":
                depth_fail += 1
            else:
                if current_phase == "width":
                    width_fail += 1
                else:
                    depth_fail += 1

        log_phase_progress(
            progress_path,
            {
                "task": task.name,
                "phase": phase_name,
                "candidate_index": candidate_idx,
                "candidate_dir": candidate_dir.name,
                "architecture": str(next_arch),
                "best_val": float(result.best_val),
                "best_epoch": int(result.best_epoch),
                "final_epoch": int(result.final_epoch),
                "best_checkpoint": str(result.best_checkpoint),
                "last_checkpoint": str(result.last_checkpoint),
                "improved_over_global": improved,
                "search_phase": current_phase,
                "width_fail": width_fail,
                "depth_fail": depth_fail,
            },
        )

        next_candidate_index += 1

        if mode in ["alt_width", "alt_depth"]:
            current_phase = "depth" if current_phase == "width" else "width"

        state.update(
            {
                "mode": mode,
                "current_phase": current_phase,
                "width_fail": width_fail,
                "depth_fail": depth_fail,
                "best_val": global_best_val,
                "candidate_index": next_candidate_index,
                "completed": False,
                "best_candidate_dir": global_best_candidate_dir.name if global_best_candidate_dir is not None else None,
                "best_checkpoint": str(global_best_checkpoint) if global_best_checkpoint is not None else None,
            }
        )
        save_phase_state(phase_root, state)

        if mode == "width_only" and width_fail >= int(cfg.patience):
            break
        if mode == "depth_only" and depth_fail >= int(cfg.patience):
            break
        if mode in ["alt_width", "alt_depth"] and width_fail >= int(cfg.patience) and depth_fail >= int(cfg.patience):
            break
        if mode == "width_to_depth" and current_phase == "depth" and depth_fail >= int(cfg.patience):
            break
        if mode == "depth_to_width" and current_phase == "width" and width_fail >= int(cfg.patience):
            break

        candidate_dirs = sorted([p for p in phase_root.iterdir() if p.is_dir() and p.name.startswith("cand_")])

    final_model = current_base_model()
    if global_best_candidate_dir is not None and global_best_checkpoint is not None and Path(global_best_checkpoint).exists():
        meta = read_json(Path(global_best_candidate_dir) / "metadata.json")
        final_model = make_model(meta["model"]["in_dim"], meta["model"]["hidden_widths"], meta["model"]["out_dim"], meta["model"]["use_bn"]).to(device)
        final_model.load_state_dict(load_checkpoint(Path(global_best_checkpoint), device)["best_state"])

    test_metrics = eval_final(final_model, task, device, reconstruct=reconstruct)
    summary = {
        "task": task.name,
        "phase": phase_name,
        "mode": mode,
        "architecture": model_signature(final_model),
        "best_val": float(global_best_val),
        "test_metrics": test_metrics,
        "best_candidate_dir": str(global_best_candidate_dir) if global_best_candidate_dir is not None else None,
        "best_checkpoint": str(global_best_checkpoint) if global_best_checkpoint is not None else None,
        "width_fail": width_fail,
        "depth_fail": depth_fail,
        "completed": True,
    }
    write_json(summary_path, summary)
    state.update({"completed": True})
    save_phase_state(phase_root, state)
    return summary


def run_task_pipeline(
    task: Task,
    task_root: Path,
    cfg: RunConfig,
    device,
    batch_controller: Optional[AdaptiveBatchController] = None,
    log: Optional[ContinuousLogger] = None,
    progress_path: Optional[Path] = None,
) -> Dict[str, Any]:
    task_summary: Dict[str, Any] = {
        "task": task.name,
        "phases": [],
        "adp_runs": [],
        "paired_stl_runs": [],
        "comparisons": [],
        "winner": None,
    }

    best_overall: Optional[Dict[str, Any]] = None
    if log is not None:
        log.log_console(f"[TASK] start {task.name}")

    for adp_phase_name, adp_mode in GOLIATH_ADP_PHASES:
        if adp_phase_name not in cfg.phases:
            continue

        if log is not None:
            log.log_console(f"[TASK:{task.name}] ADP phase start: {adp_phase_name}")

        adp_summary = run_growth_phase(
            task,
            task_root,
            cfg,
            device,
            adp_seed_hidden(),
            adp_phase_name,
            adp_mode,
            reconstruct=True,
            batch_controller=batch_controller,
        )
        task_summary["phases"].append(adp_summary)
        task_summary["adp_runs"].append(adp_summary)

        adp_arch = extract_hidden_widths(adp_summary.get("architecture"))
        stl_phase_name = f"stl_from_{adp_phase_name}"
        stl_summary = run_stl_phase(
            task,
            task_root,
            cfg,
            device,
            adp_arch,
            phase_name=stl_phase_name,
            source_phase=adp_phase_name,
            batch_controller=batch_controller,
        )
        task_summary["phases"].append(stl_summary)
        task_summary["paired_stl_runs"].append(stl_summary)

        adp_score = float(adp_summary.get("best_val", float("inf")))
        stl_score = float(stl_summary.get("best_val", float("inf")))
        winner = "adp" if adp_score <= stl_score else "stl"
        comparison = {
            "task": task.name,
            "adp_phase": adp_phase_name,
            "stl_phase": stl_phase_name,
            "adp_best_val": adp_score,
            "stl_best_val": stl_score,
            "winner": winner,
            "winner_phase": adp_phase_name if winner == "adp" else stl_phase_name,
            "winner_value": min(adp_score, stl_score),
            "adp_architecture": adp_summary.get("architecture"),
            "stl_architecture": stl_summary.get("architecture"),
        }
        task_summary["comparisons"].append(comparison)

        if best_overall is None or comparison["winner_value"] < best_overall["winner_value"]:
            best_overall = comparison

        if progress_path is not None:
            append_csv_row(
                progress_path,
                {
                    "task": task.name,
                    "phase": adp_phase_name,
                    "phase_type": adp_mode,
                    "best_val": adp_score,
                    "best_epoch": adp_summary.get("best_epoch"),
                    "final_epoch": adp_summary.get("final_epoch", adp_summary.get("best_epoch")),
                    "test_loss": (adp_summary.get("test_metrics") or {}).get("test_loss"),
                    "test_acc": (adp_summary.get("test_metrics") or {}).get("test_acc"),
                    "best_checkpoint": adp_summary.get("best_checkpoint"),
                    "candidate_dir": adp_summary.get("best_candidate_dir"),
                },
            )
            append_csv_row(
                progress_path,
                {
                    "task": task.name,
                    "phase": stl_phase_name,
                    "phase_type": "stl_refit",
                    "best_val": stl_score,
                    "best_epoch": stl_summary.get("best_epoch"),
                    "final_epoch": stl_summary.get("final_epoch", stl_summary.get("best_epoch")),
                    "test_loss": (stl_summary.get("test_metrics") or {}).get("test_loss"),
                    "test_acc": (stl_summary.get("test_metrics") or {}).get("test_acc"),
                    "best_checkpoint": stl_summary.get("checkpoint_best"),
                    "candidate_dir": stl_summary.get("candidate_dir"),
                },
            )

        task_summary["winner"] = best_overall
        write_json(task_root / "task_summary.json", task_summary)

        if log is not None:
            log.log_console(
                f"[TASK:{task.name}] ADP={adp_phase_name} best_val={adp_score:.6f} STL={stl_phase_name} best_val={stl_score:.6f} winner={winner}"
            )

    # Optional baseline STL if explicitly requested.
    if "stl" in cfg.phases:
        baseline_name = "stl_base"
        baseline_summary = run_stl_phase(
            task,
            task_root,
            cfg,
            device,
            base_stl_hidden(task, cfg),
            phase_name=baseline_name,
            source_phase=None,
            batch_controller=batch_controller,
        )
        task_summary.setdefault("baseline_stl_runs", []).append(baseline_summary)
        task_summary["phases"].append(baseline_summary)
        baseline_score = float(baseline_summary.get("best_val", float("inf")))
        baseline_comparison = {
            "task": task.name,
            "adp_phase": None,
            "stl_phase": baseline_name,
            "adp_best_val": None,
            "stl_best_val": baseline_score,
            "winner": "stl",
            "winner_phase": baseline_name,
            "winner_value": baseline_score,
            "adp_architecture": None,
            "stl_architecture": baseline_summary.get("architecture"),
        }
        task_summary["comparisons"].append(baseline_comparison)
        if best_overall is None or baseline_score < best_overall["winner_value"]:
            best_overall = baseline_comparison
        if progress_path is not None:
            append_csv_row(
                progress_path,
                {
                    "task": task.name,
                    "phase": baseline_name,
                    "phase_type": "stl_base",
                    "best_val": baseline_summary.get("best_val"),
                    "best_epoch": baseline_summary.get("best_epoch"),
                    "final_epoch": baseline_summary.get("final_epoch", baseline_summary.get("best_epoch")),
                    "test_loss": (baseline_summary.get("test_metrics") or {}).get("test_loss"),
                    "test_acc": (baseline_summary.get("test_metrics") or {}).get("test_acc"),
                    "best_checkpoint": baseline_summary.get("checkpoint_best"),
                    "candidate_dir": baseline_summary.get("candidate_dir"),
                },
            )

    task_summary["winner"] = best_overall
    write_json(task_root / "task_summary.json", task_summary)
    if log is not None:
        log.log_console(f"[TASK] done {task.name}")
    return task_summary


def run_phase_for_task(task: Task, task_root: Path, cfg: RunConfig, device, phase_name: str, batch_controller: Optional[AdaptiveBatchController] = None) -> Dict[str, Any]:
    base_hidden = base_stl_hidden(task, cfg)
    if phase_name == "stl":
        return run_stl_phase(task, task_root, cfg, device, base_hidden, batch_controller=batch_controller)
    if phase_name in [name for name, _ in GOLIATH_ADP_PHASES]:
        return run_growth_phase(
            task,
            task_root,
            cfg,
            device,
            adp_seed_hidden(),
            phase_name,
            phase_mode(phase_name) or "width_only",
            reconstruct=True,
            batch_controller=batch_controller,
        )
    raise ValueError(f"Unknown phase: {phase_name}")


def build_run_root(cfg: RunConfig) -> Path:
    if cfg.run_root:
        return Path(cfg.run_root)
    return Path(cfg.results_dir) / f"goliath_{now_stamp()}"


def main() -> None:
    p = argparse.ArgumentParser(description="Sequential ADP-first goliath runner for DAE/DNN tasks with paired STL refits")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="DAE/DNN/results")
    p.add_argument("--run-root", type=str, default=None)
    p.add_argument("--tasks", type=str, nargs="+", default=["all"])
    p.add_argument(
        "--phases",
        type=str,
        nargs="+",
        default=["ae_width_only", "ae_depth_only", "ae_width_to_depth", "ae_depth_to_width", "ae_alt_width", "ae_alt_depth"],
    )
    p.add_argument("--batch-size", type=int, default=32768, help="Global batch-size default/override. The adaptive controller will shrink this if VRAM pressure rises.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stl-width", type=int, default=128)
    p.add_argument("--stl-depth", type=int, default=2)
    p.add_argument("--alt-start-width", type=int, default=2)
    p.add_argument("--alt-start-depth", type=int, default=2)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = RunConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        run_root=args.run_root,
        tasks=tasks,
        phases=args.phases,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        seed=args.seed,
        stl_width=args.stl_width,
        stl_depth=args.stl_depth,
        alt_start_width=args.alt_start_width,
        alt_start_depth=args.alt_start_depth,
        patience=args.patience,
        delta=args.delta,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_width=args.max_width,
        max_depth=args.max_depth,
        max_neurons=args.max_neurons,
        use_bn=not args.no_bn,
        demo=args.demo,
    )

    seed_everything(cfg.seed)
    run_root = build_run_root(cfg)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(
        run_root / "run_metadata.json",
        {
            "config": asdict(cfg),
            "git_commit": git_commit(),
            "device": str(device),
            "tasks": tasks,
            "available_tasks": task_names(),
            "timestamp": now_stamp(),
            "batch_size_override": int(cfg.batch_size),
            "batch_size_policy": "override" if int(cfg.batch_size) > 0 else "per_task_default",
            "default_batch_sizes": {task_name: default_batch_size_for_task(task_name) for task_name in tasks},
        },
    )

    progress_path = run_root / "run_progress.csv"
    log = ContinuousLogger(run_root, "goliath", "sequential")
    log.log_console(f"Run root: {run_root}")
    log.log_console(f"Tasks: {tasks}")
    log.log_console(f"Phases: {cfg.phases}")
    log.log_console(f"Device: {device}")
    log.log_console(f"Git commit: {git_commit()}")

    task_objects: Dict[str, Task] = {}
    task_roots: Dict[str, Path] = {}
    task_batch_controllers: Dict[str, AdaptiveBatchController] = {}
    task_summaries: Dict[str, Dict[str, Any]] = {}

    for task_name in tasks:
        task_batch_size = batch_size_for_task(task_name, cfg.batch_size)
        task = build_task(task_name, cfg.data_dir, task_batch_size, cfg.num_workers, cfg.seed)
        refresh_task_loaders(task, task_batch_size)
        task_objects[task_name] = task
        task_root = run_root / task_name
        task_root.mkdir(parents=True, exist_ok=True)
        task_roots[task_name] = task_root
        batch_controller = AdaptiveBatchController(
            task_batch_size,
            threshold_gb=DEFAULT_VRAM_BUDGET_GB,
            poll_interval_sec=30.0,
            shrink_factor=0.75,
            state_path=task_root / "_batch_size_state.json",
        )
        batch_controller.start()
        task_batch_controllers[task_name] = batch_controller
        write_json(
            task_root / "task_metadata.json",
            {
                "task": task.name,
                "in_dim": task.in_dim,
                "out_dim": task.out_dim,
                "task_type": task.task_type,
                "extra": task.extra,
                "config": asdict(cfg),
                "batch_size": int(task_batch_size),
                "batch_size_policy": "override" if int(cfg.batch_size) > 0 else "per_task_default",
            },
        )
        task_summaries[task_name] = {"task": task.name, "phases": []}

    run_error: Optional[BaseException] = None
    try:
        for task_name in tasks:
            task = task_objects[task_name]
            task_root = task_roots[task_name]
            batch_controller = task_batch_controllers[task_name]
            log.log_console(f"=== Task start: {task_name} ===")
            refresh_task_loaders(task, batch_controller.current_batch_size)
            task_summary = run_task_pipeline(
                task,
                task_root,
                cfg,
                device,
                batch_controller=batch_controller,
                log=log,
                progress_path=progress_path,
            )
            task_summaries[task_name] = task_summary
            write_json(task_root / "task_summary.json", task_summary)
            log.log_console(f"=== Task done: {task_name} ===")
    except BaseException as exc:
        run_error = exc
    finally:
        for controller in task_batch_controllers.values():
            controller.stop()
        try:
            summary_rows: List[Dict[str, Any]] = []
            for task_name in tasks:
                if task_name not in task_summaries:
                    continue
                summary_rows.extend(build_task_summary_csv_rows(task_name, task_summaries[task_name]))
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
                "tasks": [build_task_report(task_summaries[name]) for name in tasks if name in task_summaries],
            }
            write_json(run_root / "final_report.json", final_report)
            write_text(run_root / "final_report.md", render_final_report(final_report))
            if summary_rows:
                write_csv(
                    run_root / "task_summary.csv",
                    summary_rows,
                    [
                        "task",
                        "row_type",
                        "adp_phase",
                        "adp_mode",
                        "stl_phase",
                        "adp_best_val",
                        "stl_best_val",
                        "winner",
                        "winner_phase",
                        "winner_value",
                        "adp_architecture",
                        "stl_architecture",
                    ],
                )
            log.log_console(f"Final report written to {run_root / 'final_report.json'}")
        finally:
            log.close()
    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
