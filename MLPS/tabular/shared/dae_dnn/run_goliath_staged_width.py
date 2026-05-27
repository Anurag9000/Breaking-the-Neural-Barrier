from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from DAE.DNN import adp_search as adp_search_module
from DAE.DNN import run_goliath as rg
from DAE.DNN.mlp import MLP
from DAE.DNN.train_utils import AdaptiveBatchController
from utils.adp_logging import ContinuousLogger

from MLPS.tabular.shared.dae_dnn.adp_staged_width import can_widen_staged, expand_width_staged


def is_uniform_width(model: MLP) -> bool:
    widths = [int(w) for w in model.hidden_widths]
    return bool(widths) and all(width == widths[0] for width in widths)


def can_deepen_uniform(model: MLP, cfg: rg.RunConfig) -> bool:
    return is_uniform_width(model) and rg.can_deepen(model, cfg)


def relative_improvement_pct(previous_val: Optional[float], current_val: float) -> float:
    if previous_val is None:
        return float("inf")
    denom = max(abs(float(previous_val)), 1e-12)
    return ((float(previous_val) - float(current_val)) / denom) * 100.0


def width_stage_limit_hit(cfg: rg.RunConfig, width_stage_margin_fail: int) -> bool:
    return int(cfg.width_stage_margin_patience) > 0 and width_stage_margin_fail >= int(cfg.width_stage_margin_patience)


def width_expansion_patience(cfg: rg.RunConfig) -> int:
    return int(getattr(cfg, "width_expansion_patience", 10))


def depth_expansion_patience(cfg: rg.RunConfig) -> int:
    return int(getattr(cfg, "depth_expansion_patience", 5))


def _switch_to_width_until_uniform(
    *,
    mode: str,
    current_base: MLP,
    current_phase: str,
    width_fail: int,
    depth_fail: int,
    phase_root: Path,
    state: Dict[str, Any],
) -> Tuple[str, int, int, bool]:
    if mode not in ["alt_width", "alt_depth", "width_to_depth", "depth_to_width"]:
        return current_phase, width_fail, depth_fail, False
    if current_phase != "depth":
        return current_phase, width_fail, depth_fail, False
    if is_uniform_width(current_base):
        return current_phase, width_fail, depth_fail, False
    current_phase = "width"
    depth_fail = 0
    state.update({"current_phase": current_phase, "depth_fail": depth_fail, "width_fail": width_fail, "warmup_to_uniform": True})
    rg.save_phase_state(phase_root, state)
    return current_phase, width_fail, depth_fail, True


def run_growth_phase(
    task,
    task_root: Path,
    cfg: rg.RunConfig,
    device,
    base_hidden: List[int],
    phase_name: str,
    mode: str,
    reconstruct: bool,
    batch_controller: Optional[AdaptiveBatchController] = None,
) -> Dict[str, Any]:
    phase_root = rg.phase_root_for(task_root, phase_name)
    phase_root.mkdir(parents=True, exist_ok=True)
    progress_path = phase_root / "phase_progress.csv"
    summary_path = phase_root / "phase_summary.json"
    state_file = phase_root / "search_state.json"
    had_state = state_file.exists()
    state = rg.ensure_phase_state(phase_root, mode)

    if state.get("completed", False) and summary_path.exists():
        return rg.read_json(summary_path)

    candidate_dirs = rg.list_candidate_dirs(phase_root)
    if not had_state and candidate_dirs:
        next_candidate_index = max(int(p.name.split("_")[1]) for p in candidate_dirs) + 1
    else:
        next_candidate_index = int(state.get("candidate_index", 0))
    current_phase = state.get("current_phase", "width" if mode in ["width_only", "alt_width", "width_to_depth"] else "depth")
    global_best_val = float(state.get("best_val", 1e30))
    width_fail = int(state.get("width_fail", 0))
    depth_fail = int(state.get("depth_fail", 0))
    consecutive_fail = int(state.get("consecutive_fail", 0))
    width_stage_improved = bool(state.get("width_stage_improved", False))
    width_stage_margin_fail = int(state.get("width_stage_margin_fail", 0))
    width_stage_anchor_val = state.get("width_stage_anchor_val")
    width_stage_anchor_val = None if width_stage_anchor_val is None else float(width_stage_anchor_val)
    warmup_to_uniform = bool(state.get("warmup_to_uniform", False))
    alt_consecutive_patience = max((2 * max(width_expansion_patience(cfg), depth_expansion_patience(cfg))) + 1, 10)
    global_best_candidate_dir = rg.resolve_candidate_dir(phase_root, state.get("best_candidate_dir"))
    global_best_checkpoint = Path(state["best_checkpoint"]) if state.get("best_checkpoint") else None

    if global_best_candidate_dir is None:
        completed_dirs = [p for p in candidate_dirs if rg.candidate_completed(p)]
        if completed_dirs:
            scored: List[Tuple[float, Path]] = []
            for cand in completed_dirs:
                cand_state = rg.read_json(cand / "candidate_state.json")
                scored.append((float(cand_state.get("best_val", 1e30)), cand))
            scored.sort(key=lambda item: item[0])
            global_best_val, global_best_candidate_dir = scored[0]
            global_best_checkpoint = global_best_candidate_dir / "checkpoint_best.pt"

    def current_base_model() -> MLP:
        latest = rg.latest_completed_candidate(phase_root)
        if latest is not None:
            base, _, _ = rg.load_candidate_model(latest, device)
            return base
        return (
            rg.make_reconstruction_model(task, base_hidden, cfg.use_bn).to(device)
            if reconstruct
            else rg.make_stl_model(task, base_hidden, cfg.use_bn).to(device)
        )

    incomplete = rg.incomplete_candidate(phase_root)
    if incomplete is not None:
        candidate_idx = int(incomplete.name.split("_")[1])
        candidate_model, _, _ = rg.load_candidate_model(incomplete, device)
        logger = ContinuousLogger(incomplete, f"{task.name}_{phase_name}", phase_name)
        result = rg.training_loop(
            task=task,
            model=candidate_model,
            candidate_dir=incomplete,
            cfg=cfg,
            device=device,
            logger=logger,
            reconstruct=reconstruct,
            resume=True,
            batch_controller=batch_controller,
            display_best_floor=None if global_best_val >= 1e29 else global_best_val,
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
        rg.save_phase_state(phase_root, state)
        candidate_dirs = rg.list_candidate_dirs(phase_root)

    while True:
        completed_dirs = [p for p in candidate_dirs if rg.candidate_completed(p)]
        if completed_dirs:
            latest = completed_dirs[-1]
            latest_model, _, _ = rg.load_candidate_model(latest, device)
            current_base = latest_model
        else:
            current_base = current_base_model()

        current_phase, width_fail, depth_fail, switched = _switch_to_width_until_uniform(
            mode=mode,
            current_base=current_base,
            current_phase=current_phase,
            width_fail=width_fail,
            depth_fail=depth_fail,
            phase_root=phase_root,
            state=state,
        )
        if switched:
            continue

        phase_for_candidate = current_phase
        if next_candidate_index == 0:
            next_model = current_base
            next_arch = [int(w) for w in next_model.hidden_widths]
        else:
            if mode == "width_only":
                if not can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)) or width_fail >= width_expansion_patience(cfg):
                    break
                next_model = expand_width_staged(current_base, 1, int(cfg.max_width))
                if next_model is None:
                    break
                next_arch = [int(w) for w in next_model.hidden_widths]
            elif mode == "depth_only":
                if not can_deepen_uniform(current_base, cfg) or depth_fail >= depth_expansion_patience(cfg):
                    break
                next_model = rg.expand_depth(current_base, int(cfg.max_depth))
                if next_model is None:
                    break
                next_arch = [int(w) for w in next_model.hidden_widths]
            elif mode == "alt_width":
                if current_phase == "width":
                    if is_uniform_width(current_base) and (
                        width_fail >= width_expansion_patience(cfg) or width_stage_limit_hit(cfg, width_stage_margin_fail)
                    ):
                        if not can_deepen_uniform(current_base, cfg):
                            break
                        current_phase = "depth"
                        width_fail = 0
                        width_stage_margin_fail = 0
                        state.update(
                            {
                                "current_phase": current_phase,
                                "width_fail": width_fail,
                                "width_stage_margin_fail": width_stage_margin_fail,
                            }
                        )
                        rg.save_phase_state(phase_root, state)
                        continue
                    if not can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)):
                        if not can_deepen_uniform(current_base, cfg):
                            break
                        current_phase = "depth"
                        width_fail = 0
                        width_stage_margin_fail = 0
                        state.update(
                            {
                                "current_phase": current_phase,
                                "width_fail": width_fail,
                                "width_stage_margin_fail": width_stage_margin_fail,
                            }
                        )
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_model = expand_width_staged(current_base, 1, int(cfg.max_width))
                    if next_model is None:
                        if not can_deepen_uniform(current_base, cfg):
                            break
                        current_phase = "depth"
                        width_fail = 0
                        width_stage_margin_fail = 0
                        state.update(
                            {
                                "current_phase": current_phase,
                                "width_fail": width_fail,
                                "width_stage_margin_fail": width_stage_margin_fail,
                            }
                        )
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
                else:
                    if depth_fail >= depth_expansion_patience(cfg):
                        if consecutive_fail >= alt_consecutive_patience:
                            break
                        if not can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)):
                            break
                        current_phase = "width"
                        depth_fail = 0
                        state.update({"current_phase": current_phase, "depth_fail": depth_fail})
                        rg.save_phase_state(phase_root, state)
                        continue
                    if not can_deepen_uniform(current_base, cfg):
                        if not can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)):
                            break
                        current_phase = "width"
                        depth_fail = 0
                        state.update({"current_phase": current_phase, "depth_fail": depth_fail})
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_model = rg.expand_depth(current_base, int(cfg.max_depth))
                    if next_model is None:
                        if not can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)):
                            break
                        current_phase = "width"
                        depth_fail = 0
                        state.update({"current_phase": current_phase, "depth_fail": depth_fail})
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
            elif mode == "alt_depth":
                if current_phase == "depth":
                    if depth_fail >= depth_expansion_patience(cfg):
                        if not can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)):
                            break
                        current_phase = "width"
                        depth_fail = 0
                        state.update({"current_phase": current_phase, "depth_fail": depth_fail})
                        rg.save_phase_state(phase_root, state)
                        continue
                    if not can_deepen_uniform(current_base, cfg):
                        if not can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)):
                            break
                        current_phase = "width"
                        depth_fail = 0
                        state.update({"current_phase": current_phase, "depth_fail": depth_fail})
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_model = rg.expand_depth(current_base, int(cfg.max_depth))
                    if next_model is None:
                        if not can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)):
                            break
                        current_phase = "width"
                        depth_fail = 0
                        state.update({"current_phase": current_phase, "depth_fail": depth_fail})
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
                else:
                    if is_uniform_width(current_base) and (
                        width_fail >= width_expansion_patience(cfg) or width_stage_limit_hit(cfg, width_stage_margin_fail)
                    ):
                        if consecutive_fail >= alt_consecutive_patience:
                            break
                        if not can_deepen_uniform(current_base, cfg):
                            break
                        current_phase = "depth"
                        width_fail = 0
                        width_stage_margin_fail = 0
                        state.update(
                            {
                                "current_phase": current_phase,
                                "width_fail": width_fail,
                                "width_stage_margin_fail": width_stage_margin_fail,
                            }
                        )
                        rg.save_phase_state(phase_root, state)
                        continue
                    if not can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)):
                        if not can_deepen_uniform(current_base, cfg):
                            break
                        current_phase = "depth"
                        width_fail = 0
                        width_stage_margin_fail = 0
                        state.update(
                            {
                                "current_phase": current_phase,
                                "width_fail": width_fail,
                                "width_stage_margin_fail": width_stage_margin_fail,
                            }
                        )
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_model = expand_width_staged(current_base, 1, int(cfg.max_width))
                    if next_model is None:
                        if not can_deepen_uniform(current_base, cfg):
                            break
                        current_phase = "depth"
                        width_fail = 0
                        width_stage_margin_fail = 0
                        state.update(
                            {
                                "current_phase": current_phase,
                                "width_fail": width_fail,
                                "width_stage_margin_fail": width_stage_margin_fail,
                            }
                        )
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
            elif mode == "width_to_depth":
                if current_phase == "width":
                    width_stage_margin_limit_hit = (
                        int(cfg.width_stage_margin_patience) > 0
                        and width_stage_margin_fail >= int(cfg.width_stage_margin_patience)
                    )
                    if is_uniform_width(current_base) and (width_fail >= width_expansion_patience(cfg) or width_stage_margin_limit_hit):
                        current_phase = "depth"
                        state["current_phase"] = current_phase
                        rg.save_phase_state(phase_root, state)
                        continue
                    if not can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)):
                        current_phase = "depth"
                        state.update({"current_phase": current_phase, "width_fail": width_fail})
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_model = expand_width_staged(current_base, 1, int(cfg.max_width))
                    if next_model is None:
                        current_phase = "depth"
                        state.update({"current_phase": current_phase, "width_fail": width_fail})
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
                else:
                    if depth_fail >= depth_expansion_patience(cfg):
                        break
                    if not can_deepen_uniform(current_base, cfg):
                        if can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)):
                            current_phase = "width"
                            state.update({"current_phase": current_phase, "depth_fail": depth_fail})
                            rg.save_phase_state(phase_root, state)
                            continue
                        break
                    next_model = rg.expand_depth(current_base, int(cfg.max_depth))
                    if next_model is None:
                        break
                    next_arch = [int(w) for w in next_model.hidden_widths]
            elif mode == "depth_to_width":
                if current_phase == "depth":
                    if depth_fail >= depth_expansion_patience(cfg):
                        current_phase = "width"
                        state["current_phase"] = current_phase
                        rg.save_phase_state(phase_root, state)
                        continue
                    if not can_deepen_uniform(current_base, cfg):
                        current_phase = "width"
                        state.update({"current_phase": current_phase, "depth_fail": depth_fail})
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_model = rg.expand_depth(current_base, int(cfg.max_depth))
                    if next_model is None:
                        current_phase = "width"
                        state.update({"current_phase": current_phase, "depth_fail": depth_fail})
                        rg.save_phase_state(phase_root, state)
                        continue
                    next_arch = [int(w) for w in next_model.hidden_widths]
                else:
                    if width_fail >= width_expansion_patience(cfg) and is_uniform_width(current_base):
                        break
                    if not can_widen_staged(current_base, int(cfg.max_width), int(cfg.max_neurons)):
                        break
                    next_model = expand_width_staged(current_base, 1, int(cfg.max_width))
                    if next_model is None:
                        break
                    next_arch = [int(w) for w in next_model.hidden_widths]
            else:
                raise ValueError(f"Unknown mode: {mode}")

        candidate_idx = next_candidate_index
        candidate_dir = rg.candidate_root_for(phase_root, candidate_idx, next_arch)
        logger = ContinuousLogger(candidate_dir, f"{task.name}_{phase_name}", phase_name)
        logger.log_console(
            f"[CANDIDATE] task={task.name} phase={phase_name} index={candidate_idx} "
            f"search_phase={phase_for_candidate} architecture={rg.format_architecture_for_report(next_arch)}"
        )
        rg.write_json(
            candidate_dir / "metadata.json",
            rg.phase_metadata(
                task=task,
                phase_name=phase_name,
                phase_kind=mode,
                reconstruct=reconstruct,
                model=next_model,
                cfg=cfg,
                candidate_index=candidate_idx,
                extra={"hidden_widths": next_arch, "search_phase": phase_for_candidate},
            ),
        )
        result = rg.training_loop(
            task=task,
            model=next_model.to(device),
            candidate_dir=candidate_dir,
            cfg=cfg,
            device=device,
            logger=logger,
            reconstruct=reconstruct,
            resume=True,
            batch_controller=batch_controller,
            display_best_floor=None if global_best_val >= 1e29 else global_best_val,
        )
        logger.close()

        stage_margin_pct = ""
        width_warmup_candidate = bool(
            mode in {"alt_width", "alt_depth", "width_to_depth", "depth_to_width"}
            and phase_for_candidate == "width"
            and warmup_to_uniform
        )
        improved = False
        if not width_warmup_candidate and result.best_val < (global_best_val - float(cfg.delta)):
            improved = True
            global_best_val = float(result.best_val)
            global_best_candidate_dir = candidate_dir
            global_best_checkpoint = candidate_dir / "checkpoint_best.pt"
        if phase_for_candidate == "width":
            if width_warmup_candidate:
                if is_uniform_width(next_model):
                    warmup_to_uniform = False
                    width_fail = 0
                    width_stage_margin_fail = 0
                    width_stage_improved = False
                    width_stage_anchor_val = float(global_best_val)
            else:
                width_stage_improved = width_stage_improved or improved
                if is_uniform_width(next_model):
                    stage_margin_pct = relative_improvement_pct(width_stage_anchor_val, global_best_val)
                    if width_stage_improved:
                        width_fail = 0
                        consecutive_fail = 0
                    else:
                        width_fail += 1
                        consecutive_fail += 1
                    if width_stage_anchor_val is None:
                        width_stage_margin_fail = 0
                    elif stage_margin_pct >= float(cfg.width_stage_min_improve_pct):
                        width_stage_margin_fail = 0
                    else:
                        width_stage_margin_fail += 1
                    width_stage_anchor_val = float(global_best_val)
                    width_stage_improved = False
        else:
            if mode in {"alt_width", "alt_depth", "width_to_depth", "depth_to_width"} and not is_uniform_width(next_model):
                depth_fail = 0
                warmup_to_uniform = True
                width_fail = 0
                width_stage_margin_fail = 0
                width_stage_improved = False
                width_stage_anchor_val = float(global_best_val)
            elif not is_uniform_width(next_model):
                depth_fail = 0
            elif improved:
                depth_fail = 0
                consecutive_fail = 0
            else:
                depth_fail += 1
                consecutive_fail += 1

        rg.log_phase_progress(
            progress_path,
            {
                "task": task.name,
                "phase": phase_name,
                "candidate_index": candidate_idx,
                "candidate_dir": candidate_dir.name,
                "architecture": rg.format_architecture_for_report(next_arch),
                "best_val": float(result.best_val),
                "best_epoch": int(result.best_epoch),
                "final_epoch": int(result.final_epoch),
                "best_checkpoint": str(result.best_checkpoint),
                "last_checkpoint": str(result.last_checkpoint),
                "improved_over_global": improved,
                "search_phase": phase_for_candidate,
                "width_fail": width_fail,
                "depth_fail": depth_fail,
                "consecutive_fail": consecutive_fail,
                "width_stage_improved": width_stage_improved,
                "width_stage_margin_fail": width_stage_margin_fail,
                "width_stage_anchor_val": width_stage_anchor_val,
                "width_stage_margin_pct": stage_margin_pct if phase_for_candidate == "width" and is_uniform_width(next_model) and not width_warmup_candidate else "",
                "warmup_to_uniform": warmup_to_uniform,
            },
        )

        next_candidate_index += 1

        if mode == "alt_width":
            if phase_for_candidate == "width" and is_uniform_width(next_model) and (
                width_fail >= width_expansion_patience(cfg) or width_stage_limit_hit(cfg, width_stage_margin_fail)
            ):
                current_phase = "depth"
                width_fail = 0
                width_stage_margin_fail = 0
            elif phase_for_candidate == "depth" and depth_fail >= depth_expansion_patience(cfg):
                current_phase = "width"
                depth_fail = 0
            else:
                current_phase = "width" if phase_for_candidate == "width" else "depth"
        elif mode == "alt_depth":
            if phase_for_candidate == "depth" and depth_fail >= depth_expansion_patience(cfg):
                current_phase = "width"
                depth_fail = 0
            elif phase_for_candidate == "width" and is_uniform_width(next_model) and (
                width_fail >= width_expansion_patience(cfg) or width_stage_limit_hit(cfg, width_stage_margin_fail)
            ):
                current_phase = "depth"
                width_fail = 0
                width_stage_margin_fail = 0
            else:
                current_phase = "depth" if phase_for_candidate == "depth" else "width"
        elif mode == "width_to_depth":
            if phase_for_candidate == "depth":
                current_phase = "width"
                width_fail = 0
                width_stage_margin_fail = 0
            elif not warmup_to_uniform and is_uniform_width(next_model) and (
                width_fail >= width_expansion_patience(cfg)
                or width_stage_limit_hit(cfg, width_stage_margin_fail)
            ):
                current_phase = "depth"
            else:
                current_phase = "width"
        elif mode == "depth_to_width":
            if phase_for_candidate == "width":
                current_phase = "depth" if is_uniform_width(next_model) else "width"
                depth_fail = 0 if current_phase == "depth" else depth_fail
            elif depth_fail >= depth_expansion_patience(cfg):
                current_phase = "width"

        state.update(
            {
                "mode": mode,
                "current_phase": current_phase,
                "width_fail": width_fail,
                "depth_fail": depth_fail,
                "consecutive_fail": consecutive_fail,
                "width_stage_improved": width_stage_improved,
                "width_stage_margin_fail": width_stage_margin_fail,
                "width_stage_anchor_val": width_stage_anchor_val,
                "warmup_to_uniform": warmup_to_uniform,
                "best_val": global_best_val,
                "candidate_index": next_candidate_index,
                "completed": False,
                "best_candidate_dir": global_best_candidate_dir.name if global_best_candidate_dir is not None else None,
                "best_checkpoint": str(global_best_checkpoint) if global_best_checkpoint is not None else None,
            }
        )
        rg.save_phase_state(phase_root, state)

        if mode == "width_only" and width_fail >= width_expansion_patience(cfg):
            break
        if mode == "depth_only" and depth_fail >= depth_expansion_patience(cfg):
            break
        if mode in ["alt_width", "alt_depth"] and consecutive_fail >= alt_consecutive_patience:
            break
        if mode == "width_to_depth" and depth_fail >= depth_expansion_patience(cfg):
            break
        if mode == "depth_to_width" and width_fail >= width_expansion_patience(cfg) and is_uniform_width(next_model):
            break

        candidate_dirs = rg.list_candidate_dirs(phase_root)

    if global_best_candidate_dir is None or global_best_checkpoint is None:
        raise RuntimeError(f"No completed candidates found for phase {phase_name}")

    best_model, _, best_ckpt = rg.load_candidate_model(global_best_candidate_dir, device)
    summary = {
        "task": task.name,
        "phase": phase_name,
        "mode": mode,
        "best_candidate_dir": global_best_candidate_dir.name,
        "best_checkpoint": str(global_best_checkpoint),
        "architecture": [int(w) for w in best_model.hidden_widths],
        "best_val": float(best_ckpt["best_val"]),
        "best_epoch": int(best_ckpt["best_epoch"]),
        "final_epoch": int(best_ckpt.get("epoch", best_ckpt["best_epoch"])),
        "reconstruct": reconstruct,
    }
    rg.write_json(summary_path, summary)
    state.update(
        {
            "completed": True,
            "best_candidate_dir": global_best_candidate_dir.name,
            "best_checkpoint": str(global_best_checkpoint),
            "best_val": float(best_ckpt["best_val"]),
            "candidate_index": next_candidate_index,
        }
    )
    rg.save_phase_state(phase_root, state)
    return summary


def install_staged_width_hooks() -> None:
    rg.run_growth_phase = run_growth_phase
    rg.expand_width = expand_width_staged
    rg.can_widen = lambda model, cfg: can_widen_staged(model, int(cfg.max_width), int(cfg.max_neurons))
    adp_search_module.expand_width = expand_width_staged


def main() -> None:
    install_staged_width_hooks()
    rg.main()


if __name__ == "__main__":
    main()
