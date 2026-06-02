from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from MLPS.tabular.shared.dae_dnn.mlp import MLP
from MLPS.tabular.shared.dae_dnn.train_utils import eval_epoch, train_epoch
from MLPS.tabular.shared.dae_dnn.tasks import refresh_task_loaders
from utils.adp_contract import run_module_adp
from utils.adp_logging import ContinuousLogger
from utils.adp_state import merge_state_preserving_init


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-4
    patience: int = 5
    trials_width: int = 10
    trials_depth: int = 5
    ex_k: int = 1
    max_width: int = 4096
    max_depth: int = 5
    max_neurons: int = 10_000_000
    width_stage_margin_patience: int = 5
    width_stage_min_improve_pct: float = 1.0
    min_new_layer_width: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 100_000_000
    metrics_interval: int = 5


def _resize_linear(old: nn.Linear, new_out: int, new_in: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        if old.bias is not None and new.bias is not None:
            new.bias[:r] = old.bias[:r]
    return new


def _rebuild_mlp(model: MLP, hidden_widths: List[int]) -> MLP:
    device = next(model.parameters()).device
    in_dim = model.in_dim
    out_dim = model.out_dim
    use_bn = model.use_bn

    layers: List[nn.Module] = []
    prev = in_dim
    old_linears = [m for m in model.backbone if isinstance(m, nn.Linear)]
    for width in hidden_widths:
        if old_linears:
            linear = _resize_linear(old_linears.pop(0), int(width), prev)
        else:
            linear = nn.Linear(prev, int(width)).to(device)
        layers.append(linear)
        if use_bn:
            layers.append(nn.BatchNorm1d(int(width)).to(device))
        layers.append(nn.ReLU(inplace=True))
        prev = int(width)

    new_model = MLP(in_dim=in_dim, hidden_widths=hidden_widths, out_dim=out_dim, use_bn=use_bn).to(device)
    new_model.backbone = nn.Sequential(*layers)
    new_model.head = _resize_linear(model.head, out_dim, prev)

    merged = merge_state_preserving_init(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged)
    return new_model


def expand_width(model: MLP, ex_k: int, max_width: int) -> Optional[MLP]:
    widths = [int(width) for width in model.hidden_widths]
    if not widths:
        return None
    if len(set(widths)) == 1:
        target = min(int(max_width), max(widths) + max(1, int(ex_k)))
    else:
        target = max(widths)
    new_hidden = list(widths)
    for idx, width in enumerate(new_hidden):
        if width < target:
            new_hidden[idx] = width + 1
            break
    if new_hidden == list(model.hidden_widths):
        return None
    return _rebuild_mlp(model, new_hidden)


def expand_depth(model: MLP, max_depth: int) -> Optional[MLP]:
    if not model.hidden_widths or len(model.hidden_widths) >= int(max_depth):
        return None
    widths = [int(width) for width in model.hidden_widths]
    if len(set(widths)) != 1:
        return None
    if int(widths[-1]) <= 10:
        return None
    new_hidden = list(widths) + [10]
    return _rebuild_mlp(model, new_hidden)


def total_neurons(model: MLP) -> int:
    return int(sum(int(w) for w in model.hidden_widths) + int(model.out_dim))


def model_width(model: MLP) -> int:
    return int(max(model.hidden_widths)) if model.hidden_widths else 0


def model_depth(model: MLP) -> int:
    return int(len(model.hidden_widths))


def _format_hidden_widths(hidden_widths: List[int]) -> str:
    return str([int(w) for w in hidden_widths])


def _relative_improvement_pct(previous_val: Optional[float], current_val: float) -> float:
    if previous_val is None:
        return float("inf")
    denom = max(abs(float(previous_val)), 1e-12)
    return ((float(previous_val) - float(current_val)) / denom) * 100.0


def snapshot_arch_and_state(model: MLP, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "in_dim": model.in_dim,
        "hidden_widths": list(model.hidden_widths),
        "out_dim": model.out_dim,
        "use_bn": model.use_bn,
        "state": copy.deepcopy(state),
    }


def restore_arch_and_state(model: MLP, snap: Dict[str, Any], device) -> MLP:
    new_model = MLP(
        in_dim=int(snap["in_dim"]),
        hidden_widths=snap["hidden_widths"],
        out_dim=int(snap["out_dim"]),
        use_bn=bool(snap["use_bn"]),
    ).to(device)
    new_model.load_state_dict(snap["state"])
    return new_model


def train_with_early_stopping(
    model,
    task,
    cfg: ADPConfig,
    device,
    logger: Optional[ContinuousLogger] = None,
    measure_throughput: bool = False,
    batch_controller=None,
    display_best_floor: Optional[float] = None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: Dict[str, float] = {}
    es_counter = 0
    current_batch_size = int(getattr(task.train_loader, "batch_size", 0) or 0)

    if batch_controller is not None:
        refreshed = int(batch_controller.current_batch_size)
        if refreshed > 0 and refreshed != current_batch_size:
            refresh_task_loaders(task, refreshed)
            current_batch_size = refreshed

    for epoch in range(1, int(cfg.max_epochs) + 1):
        if batch_controller is not None:
            batch_controller.maybe_poll()
            refreshed = int(batch_controller.current_batch_size)
            if refreshed > 0 and refreshed != current_batch_size:
                refresh_task_loaders(task, refreshed)
                current_batch_size = refreshed

        tr_loss, tr_acc = train_epoch(model, task.train_loader, task.loss_fn, optimizer, device, task.task_type, cfg.grad_clip)
        val_loss, val_acc, throughput = eval_epoch(
            model, task.val_loader, task.loss_fn, device, task.task_type, measure_throughput=measure_throughput
        )

        metrics: Dict[str, float] = {}
        if task.metrics_fn and (epoch == 1 or epoch % max(int(cfg.metrics_interval), 1) == 0):
            metrics = task.metrics_fn(model, task, device) or {}

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            es_counter = 0
            improved = True
        else:
            es_counter += 1
            improved = False

        display_best = best_val if display_best_floor is None else min(best_val, float(display_best_floor))
        msg = (
            f"Epoch {epoch} | architecture={_format_hidden_widths(list(model.hidden_widths))} "
            f"train_loss={tr_loss:.6f} val_loss={val_loss:.6f} best={display_best:.6f} es={es_counter}/{cfg.patience}"
        )
        if task.task_type == "classification" and tr_acc is not None and val_acc is not None:
            msg += f" train_acc={tr_acc:.4f} val_acc={val_acc:.4f}"
        if logger is not None:
            logger.log_console(msg)
            row: Dict[str, Any] = {
                "epoch": epoch,
                "width": model_width(model),
                "depth": model_depth(model),
                "neurons": total_neurons(model),
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "best_val": best_val,
                "es_counter": es_counter,
                "improved": improved,
            }
            if tr_acc is not None:
                row["train_acc"] = tr_acc
            if val_acc is not None:
                row["val_acc"] = val_acc
            if throughput is not None:
                row["throughput"] = throughput
            row.update(metrics)
            logger.log_epoch_stats(row)
        else:
            print(msg)

        if es_counter >= int(cfg.patience):
            break

    return best_val, best_state, best_metrics


def adp_search(
    model: MLP,
    task,
    cfg: ADPConfig,
    device,
    logger: Optional[ContinuousLogger] = None,
    measure_throughput: bool = False,
    batch_controller=None,
    max_candidates: Optional[int] = None,
):
    return run_module_adp(
        globals(),
        model,
        task,
        task.val_loader,
        cfg,
        device,
        results_dir=None,
        logger=logger,
        batch_controller=batch_controller,
        measure_throughput=measure_throughput,
    )
