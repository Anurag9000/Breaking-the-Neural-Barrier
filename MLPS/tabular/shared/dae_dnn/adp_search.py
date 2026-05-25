from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from DAE.DNN.mlp import MLP
from DAE.DNN.train_utils import eval_epoch, train_epoch
from DAE.DNN.tasks import refresh_task_loaders
from utils.adp_logging import ContinuousLogger
from utils.adp_state import merge_state_preserving_init


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-4
    patience: int = 10
    trials_width: int = 0
    trials_depth: int = 0
    ex_k: int = 1
    max_width: int = 4096
    max_depth: int = 10
    max_neurons: int = 10_000_000
    width_stage_margin_patience: int = 5
    width_stage_min_improve_pct: float = 1.0
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
    candidate_evals = 0

    def train_candidate(curr_model: MLP):
        nonlocal candidate_evals
        if max_candidates is not None and candidate_evals >= int(max_candidates):
            return None
        candidate_evals += 1
        return train_with_early_stopping(
            curr_model,
            task,
            cfg,
            device,
            logger,
            measure_throughput,
            batch_controller=batch_controller,
            display_best_floor=None if candidate_evals <= 1 else global_best_val,
        )

    first_run = train_candidate(model)
    if first_run is None:
        return float("inf"), model
    best_val, best_state, _ = first_run
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
            logger.log_console(f"[PHASE][WIDTH] start widths={_format_hidden_widths(list(curr_model.hidden_widths))}")

        first_local = train_candidate(curr_model)
        if first_local is None:
            return curr_model, global_best_val, global_best_snap
        local_val, local_state, _ = first_local
        local_best_val = local_val
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)

        width_fail = 0
        width_limit = int(cfg.trials_width) if int(cfg.trials_width) > 0 else max(int(cfg.patience), 1)

        while width_fail < width_limit:
            if not can_widen(curr_model):
                break
            next_model = expand_width(curr_model, cfg.ex_k, cfg.max_width)
            if next_model is None:
                break
            prev_widths = list(curr_model.hidden_widths)
            curr_model = next_model
            if logger is not None:
                logger.log_console(f"[EXPAND][WIDTH] {_format_hidden_widths(prev_widths)} -> {_format_hidden_widths(list(curr_model.hidden_widths))}")

            next_run = train_candidate(curr_model)
            if next_run is None:
                break
            v, s, _ = next_run
            if v < local_best_val - cfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                width_fail = 0
                if logger is not None:
                    logger.log_console(f"[OPT][WIDTH] improvement val_loss={v:.6f}")
            else:
                width_fail += 1
                if logger is not None:
                    logger.log_console(f"[OPT][WIDTH] no_improve val_loss={v:.6f} fail={width_fail}/{width_limit}")

        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        if logger is not None:
            logger.log_console(f"[PHASE][WIDTH] end best_val_loss={local_best_val:.6f} best_widths={_format_hidden_widths(list(final_model.hidden_widths))}")
        return final_model, local_best_val, local_best_snap

    def optimize_depth_at_fixed_width(curr_model: MLP):
        if logger is not None:
            logger.log_console(f"[PHASE][DEPTH] start widths={_format_hidden_widths(list(curr_model.hidden_widths))}")

        first_local = train_candidate(curr_model)
        if first_local is None:
            return curr_model, global_best_val, global_best_snap
        local_val, local_state, _ = first_local
        local_best_val = local_val
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)

        depth_fail = 0
        depth_limit = int(cfg.trials_depth) if int(cfg.trials_depth) > 0 else max(int(cfg.patience), 1)

        while depth_fail < depth_limit:
            if not can_deepen(curr_model):
                break
            next_model = expand_depth(curr_model, cfg.max_depth)
            if next_model is None:
                break
            prev_depth = len(curr_model.hidden_widths)
            curr_model = next_model
            if logger is not None:
                logger.log_console(f"[EXPAND][DEPTH] depth {prev_depth} -> {len(curr_model.hidden_widths)}")

            next_run = train_candidate(curr_model)
            if next_run is None:
                break
            v, s, _ = next_run
            if v < local_best_val - cfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                depth_fail = 0
                if logger is not None:
                    logger.log_console(f"[OPT][DEPTH] improvement val_loss={v:.6f}")
            else:
                depth_fail += 1
                if logger is not None:
                    logger.log_console(f"[OPT][DEPTH] no_improve val_loss={v:.6f} fail={depth_fail}/{depth_limit}")

        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        if logger is not None:
            logger.log_console(f"[PHASE][DEPTH] end best_val_loss={local_best_val:.6f} best_depth={len(final_model.hidden_widths)}")
        return final_model, local_best_val, local_best_snap

    def evaluate_candidate_model(curr_model: MLP):
        trained = train_candidate(curr_model)
        if trained is None:
            return None
        cand_val, cand_state, _ = trained
        curr_model.load_state_dict(cand_state)
        return curr_model, float(cand_val), snapshot_arch_and_state(curr_model, cand_state)

    mode = cfg.adp_mode
    current_phase = "width" if mode in ["width_only", "width", "alt_width", "width_to_depth"] else "depth"
    width_fail = 0
    depth_fail = 0
    consecutive_fail = 0
    width_stage_margin_fail = 0
    width_stage_anchor_val: Optional[float] = float(global_best_val)
    alt_consecutive_patience = max((2 * int(cfg.patience)) + 1, 10)
    current_model = restore_arch_and_state(model, global_best_snap, device)

    while True:
        phase_for_candidate = current_phase
        next_model: Optional[MLP] = None

        if mode in ["width_only", "width"]:
            if width_fail >= int(cfg.patience) or not can_widen(current_model):
                break
            next_model = expand_width(current_model, cfg.ex_k, cfg.max_width)
        elif mode in ["depth_only", "depth"]:
            if depth_fail >= int(cfg.patience) or not can_deepen(current_model):
                break
            next_model = expand_depth(current_model, cfg.max_depth)
        elif mode == "width_to_depth":
            if current_phase == "width":
                if (
                    width_fail >= int(cfg.patience)
                    or (
                        int(cfg.width_stage_margin_patience) > 0
                        and width_stage_margin_fail >= int(cfg.width_stage_margin_patience)
                    )
                ):
                    current_phase = "depth"
                    continue
                if not can_widen(current_model):
                    current_phase = "depth"
                    continue
                next_model = expand_width(current_model, cfg.ex_k, cfg.max_width)
            else:
                if depth_fail >= int(cfg.patience) or not can_deepen(current_model):
                    break
                next_model = expand_depth(current_model, cfg.max_depth)
        elif mode == "depth_to_width":
            if current_phase == "depth":
                if depth_fail >= int(cfg.patience):
                    current_phase = "width"
                    continue
                if not can_deepen(current_model):
                    current_phase = "width"
                    continue
                next_model = expand_depth(current_model, cfg.max_depth)
            else:
                if width_fail >= int(cfg.patience) or not can_widen(current_model):
                    break
                next_model = expand_width(current_model, cfg.ex_k, cfg.max_width)
        elif mode == "alt_width":
            if current_phase == "width":
                if width_fail >= int(cfg.patience):
                    if consecutive_fail >= alt_consecutive_patience:
                        break
                    if not can_deepen(current_model):
                        break
                    current_phase = "depth"
                    width_fail = 0
                    continue
                if not can_widen(current_model):
                    if not can_deepen(current_model):
                        break
                    current_phase = "depth"
                    width_fail = 0
                    continue
                next_model = expand_width(current_model, cfg.ex_k, cfg.max_width)
            else:
                if depth_fail >= int(cfg.patience):
                    if consecutive_fail >= alt_consecutive_patience:
                        break
                    if not can_widen(current_model):
                        break
                    current_phase = "width"
                    depth_fail = 0
                    continue
                if not can_deepen(current_model):
                    if not can_widen(current_model):
                        break
                    current_phase = "width"
                    depth_fail = 0
                    continue
                next_model = expand_depth(current_model, cfg.max_depth)
        elif mode == "alt_depth":
            if current_phase == "depth":
                if depth_fail >= int(cfg.patience):
                    if not can_widen(current_model):
                        break
                    current_phase = "width"
                    depth_fail = 0
                    continue
                if not can_deepen(current_model):
                    if not can_widen(current_model):
                        break
                    current_phase = "width"
                    depth_fail = 0
                    continue
                next_model = expand_depth(current_model, cfg.max_depth)
            else:
                if width_fail >= int(cfg.patience):
                    if consecutive_fail >= alt_consecutive_patience:
                        break
                    if not can_deepen(current_model):
                        break
                    current_phase = "depth"
                    width_fail = 0
                    continue
                if not can_widen(current_model):
                    if not can_deepen(current_model):
                        break
                    current_phase = "depth"
                    width_fail = 0
                    continue
                next_model = expand_width(current_model, cfg.ex_k, cfg.max_width)
        else:
            raise ValueError(f"Unknown ADP mode: {mode}")

        if next_model is None:
            break

        evaluated = evaluate_candidate_model(next_model)
        if evaluated is None:
            break
        current_model, cand_val, cand_snap = evaluated

        improved = cand_val < global_best_val - cfg.delta
        if improved:
            global_best_val = cand_val
            global_best_snap = cand_snap
            consecutive_fail = 0
            if phase_for_candidate == "width":
                width_fail = 0
            else:
                depth_fail = 0
        else:
            consecutive_fail += 1
            if phase_for_candidate == "width":
                width_fail += 1
            else:
                depth_fail += 1

        if phase_for_candidate == "width":
            stage_margin_pct = _relative_improvement_pct(width_stage_anchor_val, global_best_val)
            if width_stage_anchor_val is None:
                width_stage_margin_fail = 0
            elif stage_margin_pct >= float(cfg.width_stage_min_improve_pct):
                width_stage_margin_fail = 0
            else:
                width_stage_margin_fail += 1
            width_stage_anchor_val = float(global_best_val)

        if mode == "alt_width":
            if phase_for_candidate == "width" and (
                width_fail >= int(cfg.patience)
                or (
                    int(cfg.width_stage_margin_patience) > 0
                    and width_stage_margin_fail >= int(cfg.width_stage_margin_patience)
                )
            ):
                current_phase = "depth"
                width_fail = 0
                width_stage_margin_fail = 0
            elif phase_for_candidate == "depth" and depth_fail >= int(cfg.patience):
                current_phase = "width"
                depth_fail = 0
                width_stage_margin_fail = 0
        elif mode == "alt_depth":
            if phase_for_candidate == "depth" and depth_fail >= int(cfg.patience):
                current_phase = "width"
                depth_fail = 0
                width_stage_margin_fail = 0
            elif phase_for_candidate == "width" and (
                width_fail >= int(cfg.patience)
                or (
                    int(cfg.width_stage_margin_patience) > 0
                    and width_stage_margin_fail >= int(cfg.width_stage_margin_patience)
                )
            ):
                current_phase = "depth"
                width_fail = 0
        elif mode == "width_to_depth":
            if phase_for_candidate == "depth":
                current_phase = "width"
                width_fail = 0
                width_stage_margin_fail = 0
            elif (
                width_fail >= int(cfg.patience)
                or (
                    int(cfg.width_stage_margin_patience) > 0
                    and width_stage_margin_fail >= int(cfg.width_stage_margin_patience)
                )
            ):
                current_phase = "depth"
        elif mode == "depth_to_width":
            if phase_for_candidate == "width":
                current_phase = "depth"
                depth_fail = 0
                width_stage_margin_fail = 0
            elif depth_fail >= int(cfg.patience):
                current_phase = "width"
                width_stage_margin_fail = 0

        if mode in ["alt_width", "alt_depth"] and consecutive_fail >= alt_consecutive_patience:
            break
        if mode == "width_to_depth" and depth_fail >= int(cfg.patience):
            break
        if mode == "depth_to_width" and width_fail >= int(cfg.patience):
            break

    model = restore_arch_and_state(current_model, global_best_snap, device)
    return global_best_val, model
