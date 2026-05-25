from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


try:
    from utils.adp_introspect import infer_adp_shape
except Exception:  # pragma: no cover
    infer_adp_shape = None  # type: ignore


def _first_callable(module_globals: Dict[str, Any], names: Iterable[str]) -> Optional[Callable[..., Any]]:
    for name in names:
        fn = module_globals.get(name)
        if callable(fn):
            return fn
    return None


def _call_best_effort(fn: Callable[..., Any], pool: Dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    args = []
    kwargs: Dict[str, Any] = {}
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        value = pool.get(param.name, inspect._empty)
        if value is inspect._empty:
            if param.default is not inspect._empty:
                continue
            # Common fallbacks for modules that use generic names.
            for alias in ("model", "snap", "snapshot", "state_dict", "acfg", "cfg", "device"):
                if alias in pool:
                    value = pool[alias]
                    break
        if value is inspect._empty:
            if param.default is inspect._empty:
                args.append(None)
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[param.name] = value
        else:
            args.append(value)
    return fn(*args, **kwargs)


def _shape_from_snapshot(snapshot: Any, model: Any) -> Tuple[int, int, Optional[Tuple[int, ...]]]:
    widths = None
    if isinstance(snapshot, dict):
        width_values = snapshot.get("widths")
        if width_values is None:
            width_values = snapshot.get("hidden_widths")
        if width_values is not None:
            widths = tuple(int(w) for w in width_values)
            if widths:
                return max(widths), len(widths), widths
        width = snapshot.get("width")
        depth = snapshot.get("depth")
        if width is not None and depth is not None:
            return int(width), int(depth), widths
        arch = snapshot.get("arch")
        if isinstance(arch, dict):
            width = arch.get("width", width)
            depth = arch.get("depth", depth)
            arch_widths = arch.get("widths", arch.get("hidden_widths", widths))
            if arch_widths is not None:
                widths = tuple(int(w) for w in arch_widths)
                if widths:
                    return max(widths), len(widths), widths
            if width is not None and depth is not None:
                return int(width), int(depth), widths

    if infer_adp_shape is not None:
        try:
            width, depth = infer_adp_shape(model)
            return int(width), int(depth), widths
        except Exception:
            pass

    width = int(getattr(model, "width", getattr(model, "dim", getattr(model, "hidden_dim", 0))))
    depth = int(getattr(model, "depth", getattr(model, "_depth", len(getattr(model, "hidden_layers", [])) or 1)))
    if hasattr(model, "widths"):
        try:
            widths = tuple(int(w) for w in getattr(model, "widths"))
            if widths:
                return max(widths), len(widths), widths
        except Exception:
            pass
    return width, depth, widths


def _total_neurons(module_globals: Dict[str, Any], model: Any, width: int, depth: int, widths: Optional[Tuple[int, ...]]) -> int:
    total_fn = _first_callable(module_globals, ("total_neurons",))
    if total_fn is not None:
        try:
            return int(_call_best_effort(total_fn, {
                "model": model,
                "width": width,
                "depth": depth,
                "widths": list(widths) if widths is not None else None,
            }))
        except Exception:
            pass
    if widths is not None:
        return int(sum(widths))
    return int(width) * max(1, int(depth))


def _widths_are_uniform(widths: Optional[Tuple[int, ...]]) -> bool:
    return bool(widths) and len(set(int(w) for w in widths)) == 1


def _can_spawn_new_depth_layer(widths: Optional[Tuple[int, ...]], min_new_layer_width: int) -> bool:
    if not _widths_are_uniform(widths):
        return False
    if not widths:
        return False
    return int(widths[0]) > int(min_new_layer_width)


def _pct_improvement(prev_val: float, current_val: float) -> float:
    denom = abs(float(prev_val))
    if denom <= 1e-12:
        return 0.0 if current_val >= prev_val else float("inf")
    return ((float(prev_val) - float(current_val)) / denom) * 100.0


def _snapshot(module_globals: Dict[str, Any], model: Any) -> Any:
    snapshot_fn = _first_callable(module_globals, ("snapshot_arch_and_state", "snapshot"))
    if snapshot_fn is not None:
        try:
            return _call_best_effort(snapshot_fn, {
                "model": model,
                "state_dict": copy.deepcopy(model.state_dict()) if hasattr(model, "state_dict") else None,
            })
        except Exception:
            pass
    if hasattr(model, "state_dict"):
        return copy.deepcopy(model.state_dict())
    return copy.deepcopy(model)


def _restore(module_globals: Dict[str, Any], model: Any, snapshot: Any, device: Any = None) -> Any:
    restore_fn = _first_callable(module_globals, ("restore_arch_and_state", "restore"))
    if restore_fn is not None:
        try:
            result = _call_best_effort(restore_fn, {
                "model": model,
                "snap": snapshot,
                "snapshot": snapshot,
                "state_dict": snapshot if isinstance(snapshot, dict) else None,
                "device": device,
            })
            return model if result is None else result
        except Exception:
            pass
    if hasattr(model, "load_state_dict") and isinstance(snapshot, dict):
        model.load_state_dict(snapshot, strict=False)
    return model


def _invoke_train(
    train_fn: Callable[..., Any],
    model: Any,
    dl_train: Iterable,
    dl_val: Iterable,
    acfg: Any,
    device: Any,
    history: list,
    logger: Any = None,
    batch_controller: Any = None,
    measure_throughput: bool = False,
) -> float:
    result = _call_best_effort(train_fn, {
        "model": model,
        "local_model": model,
        "curr_model": model,
        "task": dl_train,
        "train_loader": dl_train,
        "dl_train": dl_train,
        "train_data": dl_train,
        "val_loader": dl_val,
        "dl_val": dl_val,
        "val_data": dl_val,
        "acfg": acfg,
        "cfg": acfg,
        "config": acfg,
        "device": device,
        "history": history,
        "val_history": history,
        "logger": logger,
        "batch_controller": batch_controller,
        "measure_throughput": measure_throughput,
        "verbose": True,
    })
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, (float, int)):
                return float(item)
            if hasattr(item, "item"):
                try:
                    return float(item.item())
                except Exception:
                    pass
    if hasattr(result, "item"):
        try:
            return float(result.item())
        except Exception:
            pass
    return float(result)


def _invoke_expand(
    expand_fn: Callable[..., Any],
    module_globals: Dict[str, Any],
    model: Any,
    acfg: Any,
    device: Any,
    kind: str,
) -> Any:
    pool = {
        "model": model,
        "local_model": model,
        "curr_model": model,
        "acfg": acfg,
        "cfg": acfg,
        "config": acfg,
        "device": device,
        "ex_k": getattr(acfg, "ex_k", getattr(acfg, "ex_k_width", 1)),
        "ex_k_width": getattr(acfg, "ex_k_width", getattr(acfg, "ex_k", 1)),
        "ex_k_depth": getattr(acfg, "ex_k_depth", 1),
        "max_width": getattr(acfg, "max_width", None),
        "max_depth": getattr(acfg, "max_depth", None),
        "max_neurons": getattr(acfg, "max_neurons", None),
    }
    result = _call_best_effort(expand_fn, pool)
    return model if result is None else result


def _try_expand_once(
    module_globals: Dict[str, Any],
    model: Any,
    acfg: Any,
    device: Any,
    kind: str,
    expand_fn: Optional[Callable[..., Any]],
) -> Optional[Any]:
    if expand_fn is None:
        return None
    before = _snapshot(module_globals, model)
    current_width, current_depth, current_widths = _shape_from_snapshot(before, model)
    current_total = _total_neurons(module_globals, model, current_width, current_depth, current_widths)
    next_model = _invoke_expand(expand_fn, module_globals, model, acfg, device, kind)
    new_width, new_depth, new_widths = _shape_from_snapshot(_snapshot(module_globals, next_model), next_model)
    new_total = _total_neurons(module_globals, next_model, new_width, new_depth, new_widths)
    if kind == "width" and new_total <= current_total:
        return _restore(module_globals, model, before, device)
    if kind == "depth" and new_depth <= current_depth:
        return _restore(module_globals, model, before, device)

    max_width = getattr(acfg, "max_width", getattr(acfg, "max_neurons", None))
    max_depth = getattr(acfg, "max_depth", None)
    max_neurons = getattr(acfg, "max_neurons", None)
    ex_k_width = getattr(acfg, "ex_k_width", getattr(acfg, "ex_k", 1))
    ex_k_depth = getattr(acfg, "ex_k_depth", 1)

    if kind == "width" and max_width is not None and new_width > int(max_width):
        return _restore(module_globals, model, before, device)
    if kind == "depth" and max_depth is not None and current_depth + int(ex_k_depth) > int(max_depth):
        return _restore(module_globals, model, before, device)
    if max_neurons is not None and new_total > int(max_neurons):
        return _restore(module_globals, model, before, device)
    return next_model


def run_module_adp(
    module_globals: Dict[str, Any],
    model: Any,
    dl_train: Iterable,
    dl_val: Iterable,
    acfg: Any,
    device: Any,
    *,
    log_loss: bool = False,
    log_neurons: bool = False,
    results_dir: Optional[Path] = None,
    logger: Any = None,
    batch_controller: Any = None,
    measure_throughput: bool = False,
) -> Tuple[float, Any]:
    results_dir = Path(results_dir) if results_dir is not None else Path("results_adp")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_fn = _first_callable(module_globals, ("train_with_early_stopping", "train_with_patience"))
    if train_fn is None:
        raise RuntimeError("No train_with_early_stopping/train_with_patience function found in module")

    expand_width_fn = _first_callable(module_globals, ("expand_width", "widen_all", "widen_model", "expand_model"))
    expand_depth_fn = _first_callable(module_globals, ("expand_depth", "append_depth", "deepen_model"))

    try:
        from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons
    except Exception:  # pragma: no cover
        plot_loss_vs_epoch = None  # type: ignore
        plot_loss_vs_neurons = None  # type: ignore

    val_history: list[float] = []
    improvements: list[tuple[int, float]] = []

    delta_width = float(getattr(acfg, "delta_width", getattr(acfg, "delta", 0.0) or 0.0))
    delta_depth = float(getattr(acfg, "delta_depth", getattr(acfg, "delta", 0.0) or 0.0))
    patience_width = int(getattr(acfg, "patience_width_exp", getattr(acfg, "trials_width", 10)))
    patience_depth = int(getattr(acfg, "patience_depth_exp", getattr(acfg, "trials_depth", 5)))
    width_stage_margin_patience = int(getattr(acfg, "width_stage_margin_patience", 5))
    width_stage_min_improve_pct = float(getattr(acfg, "width_stage_min_improve_pct", 1.0))
    min_new_layer_width = int(getattr(acfg, "min_new_layer_width", 10))

    def snapshot_shape(cur_model: Any) -> Tuple[int, int, Optional[Tuple[int, ...]]]:
        return _shape_from_snapshot(_snapshot(module_globals, cur_model), cur_model)

    def total_neurons(cur_model: Any) -> int:
        width, depth, widths = snapshot_shape(cur_model)
        return _total_neurons(module_globals, cur_model, width, depth, widths)

    def describe(cur_model: Any) -> str:
        _, _, widths = snapshot_shape(cur_model)
        if widths is not None:
            return str(list(widths))
        width, depth, _ = snapshot_shape(cur_model)
        return f"width={width}, depth={depth}"

    def train(cur_model: Any) -> Tuple[float, Any]:
        val = _invoke_train(
            train_fn,
            cur_model,
            dl_train,
            dl_val,
            acfg,
            device,
            val_history,
            logger=logger,
            batch_controller=batch_controller,
            measure_throughput=measure_throughput,
        )
        snap = _snapshot(module_globals, cur_model)
        return val, snap

    def restore(cur_model: Any, snap: Any) -> Any:
        return _restore(module_globals, cur_model, snap, device)

    mode = getattr(acfg, "adp_mode", "width_to_depth")

    best_val, best_snap = train(model)
    model = restore(model, best_snap)
    global_best_val = best_val
    global_best_snap = best_snap
    improvements.append((total_neurons(model), best_val))

    def update_global_best(cur_model: Any, cand_val: float, cand_snap: Any, delta: float) -> bool:
        nonlocal global_best_val, global_best_snap
        if cand_val < global_best_val - delta:
            global_best_val = cand_val
            global_best_snap = cand_snap
            improvements.append((total_neurons(cur_model), global_best_val))
            return True
        return False

    def ensure_uniform_width(cur_model: Any, *, update_global: bool = True) -> Tuple[Any, bool, Optional[float], Any]:
        progressed = False
        last_val: Optional[float] = None
        last_snap: Any = None
        while True:
            _, _, widths = snapshot_shape(cur_model)
            if _widths_are_uniform(widths):
                return cur_model, progressed, last_val, last_snap
            candidate = _try_expand_once(module_globals, cur_model, acfg, device, "width", expand_width_fn)
            if candidate is None:
                return cur_model, progressed, last_val, last_snap
            if logger is not None:
                logger.log_console(f"[STAGED][WIDTH-FILL] {describe(cur_model)} -> {describe(candidate)}")
            cand_val, cand_snap = train(candidate)
            cur_model = restore(candidate, cand_snap)
            last_val = cand_val
            last_snap = cand_snap
            if update_global:
                update_global_best(cur_model, cand_val, cand_snap, delta_width)
            progressed = True

    def run_width_stage(cur_model: Any) -> Tuple[Any, bool, bool, float]:
        stage_anchor = float(global_best_val)
        progressed = False
        while True:
            candidate = _try_expand_once(module_globals, cur_model, acfg, device, "width", expand_width_fn)
            if candidate is None:
                return cur_model, False, False, 0.0
            if logger is not None:
                logger.log_console(f"[STAGED][WIDTH] {describe(cur_model)} -> {describe(candidate)}")
            cand_val, cand_snap = train(candidate)
            cur_model = restore(candidate, cand_snap)
            improved_global = update_global_best(cur_model, cand_val, cand_snap, delta_width)
            progressed = True
            _, _, widths = snapshot_shape(cur_model)
            if _widths_are_uniform(widths):
                stage_pct = _pct_improvement(stage_anchor, global_best_val)
                return cur_model, progressed, (global_best_val < stage_anchor - delta_width), stage_pct

    def run_width_phase(cur_model: Any) -> Tuple[Any, bool]:
        width_fail = 0
        margin_fail = 0
        any_phase_improvement = False
        while True:
            before_val = float(global_best_val)
            cur_model, completed, stage_improved, stage_pct = run_width_stage(cur_model)
            if not completed:
                break
            any_phase_improvement = any_phase_improvement or stage_improved
            width_fail = 0 if stage_improved else width_fail + 1
            margin_fail = 0 if stage_pct >= width_stage_min_improve_pct else margin_fail + 1
            if logger is not None:
                logger.log_console(
                    f"[STAGED][WIDTH] completed arch={describe(cur_model)} "
                    f"stage_improved={stage_improved} stage_pct={stage_pct:.4f} "
                    f"global_best={global_best_val:.6f} width_fail={width_fail}/{patience_width} "
                    f"margin_fail={margin_fail}/{width_stage_margin_patience}"
                )
            if width_fail >= patience_width or margin_fail >= width_stage_margin_patience:
                break
            if float(global_best_val) >= before_val and not stage_improved and margin_fail >= width_stage_margin_patience:
                break
        cur_model, _, _, _ = ensure_uniform_width(cur_model)
        return cur_model, any_phase_improvement

    def run_depth_step(cur_model: Any) -> Tuple[Any, bool]:
        cur_model, _, _, _ = ensure_uniform_width(cur_model)
        _, _, widths = snapshot_shape(cur_model)
        if not _can_spawn_new_depth_layer(widths, min_new_layer_width):
            return cur_model, False
        candidate = _try_expand_once(module_globals, cur_model, acfg, device, "depth", expand_depth_fn)
        if candidate is None:
            return cur_model, False
        if logger is not None:
            logger.log_console(f"[STAGED][DEPTH] {describe(cur_model)} -> {describe(candidate)}")
        _, _, candidate_widths = snapshot_shape(candidate)
        if _widths_are_uniform(candidate_widths):
            cand_val, cand_snap = train(candidate)
            cur_model = restore(candidate, cand_snap)
            return cur_model, update_global_best(cur_model, cand_val, cand_snap, delta_depth)
        warmed_model, _, warmed_val, warmed_snap = ensure_uniform_width(candidate, update_global=False)
        if warmed_val is None or warmed_snap is None:
            return cur_model, False
        cur_model = restore(warmed_model, warmed_snap)
        return cur_model, update_global_best(cur_model, warmed_val, warmed_snap, delta_depth)

    def run_depth_phase(cur_model: Any) -> Tuple[Any, bool]:
        cur_model, _, _, _ = ensure_uniform_width(cur_model)
        depth_fail = 0
        any_phase_improvement = False
        while depth_fail < patience_depth:
            cur_model, improved = run_depth_step(cur_model)
            if logger is not None:
                logger.log_console(
                    f"[STAGED][DEPTH] arch={describe(cur_model)} improved={improved} "
                    f"global_best={global_best_val:.6f} depth_fail={depth_fail if improved else depth_fail + 1}/{patience_depth}"
                )
            if not improved:
                probe_snap = _snapshot(module_globals, cur_model)
                _, _, probe_widths = snapshot_shape(cur_model)
                if not _can_spawn_new_depth_layer(probe_widths, min_new_layer_width):
                    break
                if _try_expand_once(module_globals, cur_model, acfg, device, "depth", expand_depth_fn) is None:
                    break
                cur_model = restore(cur_model, probe_snap)
            if not improved:
                depth_fail += 1
            else:
                depth_fail = 0
                any_phase_improvement = True
        return cur_model, any_phase_improvement

    if mode in ("width_only", "width"):
        model, _ = run_width_phase(model)
    elif mode in ("depth_only", "depth"):
        model, _ = run_depth_phase(model)
    elif mode == "width_to_depth":
        depth_fail = 0
        while depth_fail < patience_depth:
            model, _ = run_width_phase(model)
            before_cycle = float(global_best_val)
            model, improved = run_depth_step(model)
            if not improved:
                depth_fail += 1
            else:
                depth_fail = 0
            if float(global_best_val) >= before_cycle and not improved:
                break
    elif mode == "depth_to_width":
        width_fail = 0
        while width_fail < patience_width:
            model, _ = run_depth_phase(model)
            before_cycle = float(global_best_val)
            model, width_improved = run_width_phase(model)
            if float(global_best_val) < before_cycle - delta_width or width_improved:
                width_fail = 0
            else:
                width_fail += 1
    elif mode == "alt_width":
        width_done = False
        depth_done = False
        phase = "width"
        while not (width_done and depth_done):
            if phase == "width":
                before = float(global_best_val)
                model, improved = run_width_phase(model)
                width_done = not (improved or float(global_best_val) < before - delta_width)
                phase = "depth"
            else:
                before = float(global_best_val)
                model, improved = run_depth_phase(model)
                depth_done = not (improved or float(global_best_val) < before - delta_depth)
                phase = "width"
    elif mode == "alt_depth":
        width_done = False
        depth_done = False
        phase = "depth"
        while not (width_done and depth_done):
            if phase == "depth":
                before = float(global_best_val)
                model, improved = run_depth_phase(model)
                depth_done = not (improved or float(global_best_val) < before - delta_depth)
                phase = "width"
            else:
                before = float(global_best_val)
                model, improved = run_width_phase(model)
                width_done = not (improved or float(global_best_val) < before - delta_width)
                phase = "depth"
    else:
        raise ValueError(f"Unsupported ADP mode: {mode}")

    model = restore(model, global_best_snap)
    if log_loss and plot_loss_vs_epoch is not None:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{module_globals.get('__name__', 'adp')} ({mode})")
    if log_neurons and improvements and plot_loss_vs_neurons is not None:
        plot_loss_vs_neurons([n for n, _ in improvements], [v for _, v in improvements], results_dir / "loss_vs_neurons.png", title=f"{module_globals.get('__name__', 'adp')} ({mode})")

    return float(global_best_val), model
