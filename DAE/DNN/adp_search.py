from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from DAE.DNN.mlp import MLP
from DAE.DNN.train_utils import eval_epoch, train_epoch
from utils.adp_logging import ContinuousLogger
from utils.adp_state import merge_state_preserving_init


@dataclass
class ADPConfig:
    adp_mode: str = "width_only"
    delta: float = 1e-4
    patience: int = 10
    trials_width: int = 0
    trials_depth: int = 0
    ex_k: int = 1
    max_width: int = 4096
    max_depth: int = 10
    max_neurons: int = 10_000_000
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
    new_hidden = [min(int(max_width), int(width) + int(ex_k)) for width in model.hidden_widths]
    if new_hidden == list(model.hidden_widths):
        return None
    return _rebuild_mlp(model, new_hidden)


def expand_depth(model: MLP, max_depth: int) -> Optional[MLP]:
    if not model.hidden_widths or len(model.hidden_widths) >= int(max_depth):
        return None
    new_hidden = list(model.hidden_widths) + [int(model.hidden_widths[-1])]
    return _rebuild_mlp(model, new_hidden)


def total_neurons(model: MLP) -> int:
    return int(sum(int(w) for w in model.hidden_widths) + int(model.out_dim))


def model_width(model: MLP) -> int:
    return int(max(model.hidden_widths)) if model.hidden_widths else 0


def model_depth(model: MLP) -> int:
    return int(len(model.hidden_widths))


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
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: Dict[str, float] = {}
    es_counter = 0

    for epoch in range(1, int(cfg.max_epochs) + 1):
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

        msg = f"Epoch {epoch} | train_loss={tr_loss:.6f} val_loss={val_loss:.6f} best={best_val:.6f} es={es_counter}/{cfg.patience}"
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
):
    best_val, best_state, _ = train_with_early_stopping(model, task, cfg, device, logger, measure_throughput)
    model.load_state_dict(best_state)

    global_best_val = best_val
    global_best_snap = snapshot_arch_and_state(model, best_state)

    def can_widen(m: MLP) -> bool:
        if not m.hidden_widths:
            return False
        return model_width(m) + int(cfg.ex_k) <= int(cfg.max_width) and total_neurons(m) < int(cfg.max_neurons)

    def can_deepen(m: MLP) -> bool:
        if not m.hidden_widths:
            return False
        return len(m.hidden_widths) + 1 <= int(cfg.max_depth) and (total_neurons(m) + int(m.hidden_widths[-1])) <= int(cfg.max_neurons)

    def optimize_width_at_fixed_depth(curr_model: MLP):
        if logger is not None:
            logger.log_console(f"[PHASE][WIDTH] start widths={curr_model.hidden_widths}")

        local_val, local_state, _ = train_with_early_stopping(curr_model, task, cfg, device, logger, measure_throughput)
        local_best_val = local_val
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)

        width_fail = 0
        width_limit = None if int(cfg.trials_width) <= 0 else int(cfg.trials_width)

        while width_limit is None or width_fail < width_limit:
            if not can_widen(curr_model):
                break
            next_model = expand_width(curr_model, cfg.ex_k, cfg.max_width)
            if next_model is None:
                break
            prev_widths = list(curr_model.hidden_widths)
            curr_model = next_model
            if logger is not None:
                logger.log_console(f"[EXPAND][WIDTH] {prev_widths} -> {curr_model.hidden_widths}")

            v, s, _ = train_with_early_stopping(curr_model, task, cfg, device, logger, measure_throughput)
            if v < local_best_val - cfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                width_fail = 0
                if logger is not None:
                    logger.log_console(f"[OPT][WIDTH] improvement val_loss={v:.6f}")
            else:
                width_fail += 1
                if logger is not None:
                    logger.log_console(f"[OPT][WIDTH] no_improve val_loss={v:.6f} fail={width_fail}/{width_limit if width_limit is not None else 'inf'}")

        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        if logger is not None:
            logger.log_console(f"[PHASE][WIDTH] end best_val_loss={local_best_val:.6f} best_widths={final_model.hidden_widths}")
        return final_model, local_best_val, local_best_snap

    def optimize_depth_at_fixed_width(curr_model: MLP):
        if logger is not None:
            logger.log_console(f"[PHASE][DEPTH] start widths={curr_model.hidden_widths}")

        local_val, local_state, _ = train_with_early_stopping(curr_model, task, cfg, device, logger, measure_throughput)
        local_best_val = local_val
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)

        depth_fail = 0
        depth_limit = None if int(cfg.trials_depth) <= 0 else int(cfg.trials_depth)

        while depth_limit is None or depth_fail < depth_limit:
            if not can_deepen(curr_model):
                break
            next_model = expand_depth(curr_model, cfg.max_depth)
            if next_model is None:
                break
            prev_depth = len(curr_model.hidden_widths)
            curr_model = next_model
            if logger is not None:
                logger.log_console(f"[EXPAND][DEPTH] depth {prev_depth} -> {len(curr_model.hidden_widths)}")

            v, s, _ = train_with_early_stopping(curr_model, task, cfg, device, logger, measure_throughput)
            if v < local_best_val - cfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                depth_fail = 0
                if logger is not None:
                    logger.log_console(f"[OPT][DEPTH] improvement val_loss={v:.6f}")
            else:
                depth_fail += 1
                if logger is not None:
                    logger.log_console(f"[OPT][DEPTH] no_improve val_loss={v:.6f} fail={depth_fail}/{depth_limit if depth_limit is not None else 'inf'}")

        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        if logger is not None:
            logger.log_console(f"[PHASE][DEPTH] end best_val_loss={local_best_val:.6f} best_depth={len(final_model.hidden_widths)}")
        return final_model, local_best_val, local_best_snap

    mode = cfg.adp_mode
    if mode in ["width_only", "width"]:
        model, global_best_val, global_best_snap = optimize_width_at_fixed_depth(model)
    elif mode in ["depth_only", "depth"]:
        model, global_best_val, global_best_snap = optimize_depth_at_fixed_width(model)
    elif mode == "width_to_depth":
        model, global_best_val, global_best_snap = optimize_width_at_fixed_depth(model)
        width_fail = 0
        width_limit = None if int(cfg.trials_width) <= 0 else int(cfg.trials_width)
        while width_limit is None or width_fail < width_limit:
            if not can_deepen(model):
                break
            next_model = expand_depth(model, cfg.max_depth)
            if next_model is None:
                break
            model = next_model
            model, val_d, snap_d = optimize_depth_at_fixed_width(model)
            if val_d < global_best_val - cfg.delta:
                global_best_val = val_d
                global_best_snap = snap_d
                width_fail = 0
            else:
                width_fail += 1
        model = restore_arch_and_state(model, global_best_snap, device)
    elif mode == "depth_to_width":
        model, global_best_val, global_best_snap = optimize_depth_at_fixed_width(model)
        depth_fail = 0
        depth_limit = None if int(cfg.trials_depth) <= 0 else int(cfg.trials_depth)
        while depth_limit is None or depth_fail < depth_limit:
            if not can_widen(model):
                break
            next_model = expand_width(model, cfg.ex_k, cfg.max_width)
            if next_model is None:
                break
            model = next_model
            model, val_w, snap_w = optimize_width_at_fixed_depth(model)
            if val_w < global_best_val - cfg.delta:
                global_best_val = val_w
                global_best_snap = snap_w
                depth_fail = 0
            else:
                depth_fail += 1
        model = restore_arch_and_state(model, global_best_snap, device)
    elif mode in ["alt_width", "alt_depth"]:
        phase = "width" if mode == "alt_width" else "depth"
        sat_w = False
        sat_d = False
        while not (sat_w and sat_d):
            improved = False
            if phase == "width":
                model, val, snap = optimize_width_at_fixed_depth(model)
                if val < global_best_val - cfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved = True
                sat_w = not improved
                phase = "depth"
            else:
                model, val, snap = optimize_depth_at_fixed_width(model)
                if val < global_best_val - cfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved = True
                sat_d = not improved
                phase = "width"
            model = restore_arch_and_state(model, global_best_snap, device)
        model = restore_arch_and_state(model, global_best_snap, device)

    return global_best_val, model
