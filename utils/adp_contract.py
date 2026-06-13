from __future__ import annotations

import csv
import copy
import inspect
import json
import os
from pathlib import Path
import random
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch


try:
    from utils.adp_introspect import infer_adp_shape
except Exception:  # pragma: no cover
    infer_adp_shape = None  # type: ignore

try:
    from utils.adp_logging import ContinuousLogger
except Exception:  # pragma: no cover
    ContinuousLogger = None  # type: ignore


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


def _slug_architecture(width: int, depth: int) -> str:
    return f"d{int(depth)}_w{int(width)}"


def _candidate_slug(candidate_index: int, width: int, depth: int) -> str:
    return f"cand_{int(candidate_index):03d}_{_slug_architecture(width, depth)}"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _load_checkpoint(path: Path, device: Any = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _config_snapshot(config: Any) -> Dict[str, Any]:
    if hasattr(config, "__dict__"):
        return {str(key): _json_safe(value) for key, value in vars(config).items()}
    return {"repr": repr(config)}


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _tupleify(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleify(item) for item in value)
    return value


def _as_byte_tensor(value: Any) -> Any:
    if value is None:
        return value
    if torch.is_tensor(value):
        try:
            return value.detach().to(device="cpu", dtype=torch.uint8)
        except Exception:
            return value
    try:
        return torch.tensor(value, dtype=torch.uint8)
    except Exception:
        return value


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(_tupleify(state["python"]))
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "numpy" in state:
        try:
            import numpy as np

            numpy_state = _tupleify(state["numpy"])
            if isinstance(numpy_state, tuple) and len(numpy_state) >= 2 and not isinstance(numpy_state[1], np.ndarray):
                numpy_state = (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
            np.random.set_state(numpy_state)
        except Exception:
            pass
    if "cuda" in state and torch.cuda.is_available():
        cuda_state = state["cuda"]
        if isinstance(cuda_state, (list, tuple)):
            cuda_state = [_as_byte_tensor(item) for item in cuda_state]
        torch.cuda.set_rng_state_all(cuda_state)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: Any) -> None:
    for values in optimizer.state.values():
        for key, value in values.items():
            if torch.is_tensor(value):
                values[key] = value.to(device)


class _AdamWResumeCapture:
    def __init__(self, previous_state: Optional[Dict[str, Any]], device: Any) -> None:
        self.previous_state = previous_state
        self.device = device
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._original = torch.optim.AdamW

    def __enter__(self) -> "_AdamWResumeCapture":
        def factory(*args: Any, **kwargs: Any) -> torch.optim.Optimizer:
            optimizer = self._original(*args, **kwargs)
            if self.previous_state is not None:
                optimizer.load_state_dict(self.previous_state)
                _optimizer_to_device(optimizer, self.device)
            self.optimizer = optimizer
            return optimizer

        torch.optim.AdamW = factory  # type: ignore[assignment]
        return self

    def __exit__(self, *_: Any) -> None:
        torch.optim.AdamW = self._original  # type: ignore[assignment]

    def state_dict(self) -> Optional[Dict[str, Any]]:
        return copy.deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None


_OPTIMIZER_CAPTURE_LOCK = threading.RLock()


def _list_candidate_dirs(results_dir: Path) -> list[Path]:
    return sorted(
        [path for path in results_dir.iterdir() if path.is_dir() and path.name.startswith("cand_")],
        key=lambda path: path.name,
    )


def _latest_completed_candidate(results_dir: Path) -> Optional[Path]:
    for candidate_dir in reversed(_list_candidate_dirs(results_dir)):
        state = _load_json(candidate_dir / "candidate_state.json")
        if state is not None and bool(state.get("completed", False)):
            return candidate_dir
    return None


def _latest_incomplete_candidate(results_dir: Path) -> Optional[Path]:
    for candidate_dir in reversed(_list_candidate_dirs(results_dir)):
        state = _load_json(candidate_dir / "candidate_state.json")
        if state is not None and not bool(state.get("completed", False)):
            return candidate_dir
    return None


def _candidate_index_from_dir(candidate_dir: Path) -> int:
    try:
        return int(candidate_dir.name.split("_", 2)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid candidate directory name: {candidate_dir.name}") from exc


def _resolve_candidate_dir(results_dir: Path, candidate_ref: Optional[str]) -> Optional[Path]:
    if not candidate_ref:
        return None
    candidate_path = Path(candidate_ref)
    if candidate_path.exists():
        return candidate_path
    fallback = results_dir / candidate_ref
    return fallback if fallback.exists() else None


def _format_architecture(widths: Optional[Tuple[int, ...]], width: int, depth: int) -> str:
    if widths is not None:
        return str([int(item) for item in widths])
    return f"[{', '.join([str(int(width))] * max(1, int(depth)))}]"


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
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    pool = {
        "model": model,
        "local_model": model,
        "curr_model": model,
        "task": dl_train,
        "data": dl_train,
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
    }
    if overrides:
        pool.update(overrides)
    return _call_best_effort(train_fn, pool)


def _extract_train_value(result: Any) -> Tuple[float, Optional[Dict[str, Any]]]:
    if isinstance(result, tuple):
        best_state: Optional[Dict[str, Any]] = None
        numeric_value: Optional[float] = None
        for item in result:
            if isinstance(item, dict) and best_state is None:
                best_state = item
            elif numeric_value is None and isinstance(item, (float, int)):
                numeric_value = float(item)
            elif numeric_value is None and hasattr(item, "item"):
                try:
                    numeric_value = float(item.item())
                except Exception:
                    pass
        if numeric_value is None:
            numeric_value = 0.0
        return numeric_value, best_state
    if hasattr(result, "item"):
        try:
            return float(result.item()), None
        except Exception:
            pass
    if isinstance(result, (float, int)):
        return float(result), None
    return float(result), None


def _snapshot_from_state(module_globals: Dict[str, Any], model: Any, best_state: Optional[Dict[str, Any]]) -> Any:
    if best_state is None:
        return _snapshot(module_globals, model)
    snapshot_fn = _first_callable(module_globals, ("snapshot_arch_and_state", "snapshot"))
    if snapshot_fn is not None:
        try:
            return _call_best_effort(snapshot_fn, {"model": model, "state_dict": best_state})
        except Exception:
            pass
    snap = _snapshot(module_globals, model)
    if isinstance(snap, dict):
        snap = copy.deepcopy(snap)
        snap["state"] = copy.deepcopy(best_state)
    return snap


def _run_resumable_candidate_training(
    *,
    train_fn: Callable[..., Any],
    module_globals: Dict[str, Any],
    model: Any,
    dl_train: Iterable,
    dl_val: Iterable,
    acfg: Any,
    device: Any,
    logger: Any,
    batch_controller: Any,
    measure_throughput: bool,
    checkpoint_last: Path,
    checkpoint_best: Path,
    metadata: Dict[str, Any],
    train_overrides: Optional[Dict[str, Any]] = None,
    initial_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Any, list[float], int, int]:
    payload = initial_payload or {}
    completed_epochs = int(payload.get("completed_epochs", payload.get("final_epoch", 0)))
    history = [float(item) for item in payload.get("history", [])]
    best_val = float(payload.get("best_val", float("inf")))
    best_epoch = int(payload.get("best_epoch", 0))
    best_snapshot = payload.get("best_snapshot")
    es_counter = int(payload.get("es_counter", 0))
    optimizer_state = payload.get("optimizer_state")
    scheduler_state = payload.get("scheduler_state")
    del scheduler_state  # Reserved for wrappers that add schedulers later.

    if payload.get("model_state") is not None and hasattr(model, "load_state_dict"):
        try:
            model.load_state_dict(payload["model_state"], strict=False)
        except RuntimeError as exc:
            message = str(exc)
            if "size mismatch" not in message:
                raise
            if logger is not None and hasattr(logger, "log_console"):
                logger.log_console(
                    "[RESUME] incompatible checkpoint shape detected; "
                    "restarting candidate from the current model state"
                )
            payload = {}
            completed_epochs = 0
            history = []
            best_val = float("inf")
            best_epoch = 0
            best_snapshot = None
            es_counter = 0
            optimizer_state = None
    _restore_rng_state(payload.get("rng_state"))

    max_epochs = int(getattr(acfg, "max_epochs", 1))
    patience = int(getattr(acfg, "patience", 1))
    delta = float(getattr(acfg, "delta", 0.0) or 0.0)

    while completed_epochs < max_epochs and es_counter < patience:
        epoch_config = copy.copy(acfg)
        if hasattr(epoch_config, "max_epochs"):
            setattr(epoch_config, "max_epochs", 1)
        epoch_history: list[float] = []
        with _OPTIMIZER_CAPTURE_LOCK:
            with _AdamWResumeCapture(optimizer_state, device) as optimizer_capture:
                raw_result = _invoke_train(
                    train_fn,
                    model,
                    dl_train,
                    dl_val,
                    epoch_config,
                    device,
                    epoch_history,
                    logger=logger,
                    batch_controller=batch_controller,
                    measure_throughput=measure_throughput,
                    overrides={
                        **(train_overrides or {}),
                        "max_epochs": 1,
                        "patience": max(1, patience),
                    },
                )
                optimizer_state = optimizer_capture.state_dict()

        epoch_val, epoch_best_state = _extract_train_value(raw_result)
        completed_epochs += 1
        if epoch_history:
            epoch_val = float(epoch_history[-1])
            history.extend(float(item) for item in epoch_history)
        else:
            history.append(float(epoch_val))

        improved = float(epoch_val) < float(best_val) - delta
        if improved or best_snapshot is None:
            best_val = float(epoch_val)
            best_epoch = int(completed_epochs)
            best_snapshot = _snapshot_from_state(module_globals, model, epoch_best_state)
            es_counter = 0
        else:
            es_counter += 1

        epoch_payload = {
            "version": 2,
            "completed": False,
            "completed_epochs": int(completed_epochs),
            "final_epoch": int(completed_epochs),
            "best_epoch": int(best_epoch),
            "best_val": float(best_val),
            "es_counter": int(es_counter),
            "best_snapshot": best_snapshot,
            "model_state": copy.deepcopy(model.state_dict()) if hasattr(model, "state_dict") else None,
            "optimizer_state": optimizer_state,
            "scheduler_state": None,
            "rng_state": _capture_rng_state(),
            "history": history,
            "hyperparameters": _config_snapshot(acfg),
            "training_overrides": _json_safe(train_overrides or {}),
            "metadata": metadata,
        }
        _save_checkpoint(checkpoint_last, epoch_payload)
        if improved or not checkpoint_best.exists():
            _save_checkpoint(checkpoint_best, epoch_payload)

    if best_snapshot is None:
        best_snapshot = _snapshot(module_globals, model)
    return float(best_val), best_snapshot, history, int(best_epoch), int(completed_epochs)


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
    return _call_best_effort(expand_fn, pool)


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
    if next_model is None:
        return None
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
    train_overrides: Optional[Dict[str, Any]] = None,
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

    mode = getattr(acfg, "adp_mode", "width_to_depth")
    module_name = str(module_globals.get("__name__", "adp"))
    search_state_path = results_dir / "search_state.json"
    phase_progress_path = results_dir / "phase_progress.csv"
    summary_path = results_dir / "phase_summary.json"
    initial_state = _load_json(search_state_path) or {}
    created_phase_logger = False
    phase_logger = logger
    if phase_logger is None and ContinuousLogger is not None:
        phase_logger = ContinuousLogger(results_dir, module_name, mode, resume=search_state_path.exists())
        created_phase_logger = True

    val_history: list[float] = []
    improvements: list[tuple[int, float]] = []

    delta_width = float(getattr(acfg, "delta_width", getattr(acfg, "delta", 0.0) or 0.0))
    delta_depth = float(getattr(acfg, "delta_depth", getattr(acfg, "delta", 0.0) or 0.0))
    patience_width = int(
        getattr(
            acfg,
            "width_expansion_patience",
            getattr(acfg, "patience_width_exp", getattr(acfg, "trials_width", 10)),
        )
    )
    patience_depth = int(
        getattr(
            acfg,
            "depth_expansion_patience",
            getattr(acfg, "patience_depth_exp", getattr(acfg, "trials_depth", 2)),
        )
    )
    width_stage_margin_patience = int(getattr(acfg, "width_stage_margin_patience", 10))
    width_stage_min_improve_pct = float(getattr(acfg, "width_stage_min_improve_pct", 1.0))
    depth_stage_margin_patience = int(getattr(acfg, "depth_stage_margin_patience", patience_depth))
    depth_stage_min_improve_pct = float(getattr(acfg, "depth_stage_min_improve_pct", 1.0))
    min_new_layer_width = int(getattr(acfg, "min_new_layer_width", 1))
    depth_first_seed_width = int(getattr(acfg, "depth_first_seed_width", 1))

    def save_search_state(payload: Dict[str, Any]) -> None:
        _write_json(search_state_path, payload)

    def record_phase_progress(row: Dict[str, Any]) -> None:
        _append_csv_row(phase_progress_path, row)

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

    def load_candidate_model(candidate_dir: Path, checkpoint_name: str = "checkpoint_best.pt") -> Tuple[Any, Dict[str, Any]]:
        payload = _load_checkpoint(candidate_dir / checkpoint_name, device=device)
        restored_model = _restore(module_globals, model, payload["best_snapshot"], device)
        return restored_model, payload

    def recover_best_state() -> Tuple[float, Optional[Path], Optional[Path]]:
        best_value = float("inf")
        best_dir: Optional[Path] = None
        best_ckpt: Optional[Path] = None
        for candidate_dir in _list_candidate_dirs(results_dir):
            state = _load_json(candidate_dir / "candidate_state.json")
            if state is None or not bool(state.get("completed", False)):
                continue
            candidate_best = float(state.get("best_val", float("inf")))
            checkpoint = candidate_dir / "checkpoint_best.pt"
            if not checkpoint.exists():
                checkpoint = candidate_dir / "checkpoint_last.pt"
            if candidate_best < best_value and checkpoint.exists():
                best_value = candidate_best
                best_dir = candidate_dir
                best_ckpt = checkpoint
        return best_value, best_dir, best_ckpt

    def train(cur_model: Any, *, search_phase: str) -> Tuple[float, Any, Path, int, int]:
        nonlocal candidate_index
        width, depth, widths = snapshot_shape(cur_model)
        candidate_dir = results_dir / _candidate_slug(candidate_index, width, depth)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_state_path = candidate_dir / "candidate_state.json"
        metadata_path = candidate_dir / "metadata.json"
        best_ckpt = candidate_dir / "checkpoint_best.pt"
        last_ckpt = candidate_dir / "checkpoint_last.pt"
        candidate_history: list[float] = []
        candidate_logger = None
        created_candidate_logger = False
        if ContinuousLogger is not None:
            candidate_logger = ContinuousLogger(candidate_dir, module_name, mode, resume=last_ckpt.exists())
            created_candidate_logger = True

        metadata = {
            "module": module_name,
            "mode": mode,
            "candidate_index": int(candidate_index),
            "search_phase": search_phase,
            "architecture": widths if widths is not None else [int(width)] * max(1, int(depth)),
            "width": int(width),
            "depth": int(depth),
            "results_dir": str(results_dir),
            "hyperparameters": _config_snapshot(acfg),
            "resume_granularity": "last_completed_epoch",
        }
        _write_json(metadata_path, metadata)
        _write_json(
            candidate_state_path,
            {
                "completed": False,
                "candidate_dir": str(candidate_dir),
                "candidate_index": int(candidate_index),
                "search_phase": search_phase,
                "architecture": widths if widths is not None else [int(width)] * max(1, int(depth)),
                "checkpoint_best": str(best_ckpt),
                "checkpoint_last": str(last_ckpt),
            },
        )
        initial_payload = _load_checkpoint(last_ckpt, device=device) if last_ckpt.exists() else None
        try:
            val, snap, candidate_history, best_epoch, final_epoch = _run_resumable_candidate_training(
                train_fn=train_fn,
                module_globals=module_globals,
                model=cur_model,
                dl_train=dl_train,
                dl_val=dl_val,
                acfg=acfg,
                device=device,
                logger=candidate_logger or phase_logger,
                batch_controller=batch_controller,
                measure_throughput=measure_throughput,
                checkpoint_last=last_ckpt,
                checkpoint_best=best_ckpt,
                metadata={
                    "module": module_name,
                    "mode": mode,
                    "candidate_index": int(candidate_index),
                    "search_phase": search_phase,
                },
                train_overrides=train_overrides,
                initial_payload=initial_payload,
            )
        except BaseException:
            if created_candidate_logger and candidate_logger is not None:
                candidate_logger.close()
            if created_phase_logger and phase_logger is not None:
                phase_logger.close()
            raise
        cur_model = _restore(module_globals, cur_model, snap, device)
        val_history.extend(float(item) for item in candidate_history)
        checkpoint_payload = _load_checkpoint(last_ckpt, device="cpu")
        checkpoint_payload["completed"] = True
        checkpoint_payload["best_snapshot"] = snap
        _save_checkpoint(last_ckpt, checkpoint_payload)
        _write_json(
            candidate_state_path,
            {
                "completed": True,
                "candidate_dir": str(candidate_dir),
                "candidate_index": int(candidate_index),
                "search_phase": search_phase,
                "best_val": float(val),
                "best_epoch": int(best_epoch),
                "final_epoch": int(final_epoch),
                "es_counter": int(checkpoint_payload.get("es_counter", 0)),
                "architecture": widths if widths is not None else [int(width)] * max(1, int(depth)),
                "checkpoint_best": str(best_ckpt),
                "checkpoint_last": str(last_ckpt),
                "resume_granularity": "last_completed_epoch",
            },
        )
        if created_candidate_logger and candidate_logger is not None:
            candidate_logger.close()
        candidate_index += 1
        return val, snap, candidate_dir, best_epoch, final_epoch

    def restore(cur_model: Any, snap: Any) -> Any:
        return _restore(module_globals, cur_model, snap, device)

    def align_depth_first_seed(cur_model: Any) -> Any:
        if mode not in ("depth_only", "depth", "alt_depth", "depth_to_width"):
            return cur_model
        while True:
            _, _, widths = snapshot_shape(cur_model)
            if not widths:
                return cur_model
            if len(widths) != 1:
                return cur_model
            if int(widths[0]) >= depth_first_seed_width:
                return cur_model
            candidate = _try_expand_once(module_globals, cur_model, acfg, device, "width", expand_width_fn)
            if candidate is None:
                return cur_model
            cur_model = candidate

    recovered_best_val, recovered_best_dir, recovered_best_ckpt = recover_best_state()
    state_best_dir = _resolve_candidate_dir(results_dir, initial_state.get("best_candidate_dir"))
    state_best_ckpt = Path(initial_state["best_checkpoint"]) if initial_state.get("best_checkpoint") else None
    best_candidate_dir = recovered_best_dir if recovered_best_dir is not None else state_best_dir
    best_checkpoint = recovered_best_ckpt if recovered_best_ckpt is not None else state_best_ckpt
    global_best_val = float(initial_state.get("best_val", recovered_best_val if recovered_best_dir is not None else float("inf")))
    if recovered_best_dir is not None and recovered_best_val <= global_best_val:
        global_best_val = float(recovered_best_val)
    completed_candidate = _latest_completed_candidate(results_dir)
    if completed_candidate is not None:
        model, _ = load_candidate_model(completed_candidate)
    else:
        model = align_depth_first_seed(model)

    existing_candidates = _list_candidate_dirs(results_dir)
    incomplete_candidate = _latest_incomplete_candidate(results_dir)
    candidate_index = int(
        initial_state.get(
            "candidate_index",
            _candidate_index_from_dir(incomplete_candidate) if incomplete_candidate is not None else len(existing_candidates),
        )
    )
    current_phase = str(
        initial_state.get(
            "current_phase",
            "width" if mode in ("width_only", "width", "alt_width", "width_to_depth") else "depth",
        )
    )
    width_fail = int(initial_state.get("width_fail", 0))
    depth_fail = int(initial_state.get("depth_fail", 0))
    width_stage_margin_fail = int(initial_state.get("width_stage_margin_fail", 0))

    if best_candidate_dir is not None and best_checkpoint is not None and best_checkpoint.exists():
        best_model, _ = load_candidate_model(best_candidate_dir, best_checkpoint.name)
        global_best_snap = _snapshot(module_globals, best_model)
        improvements.append((total_neurons(best_model), float(global_best_val)))
        if bool(initial_state.get("completed", False)):
            final_model = restore(model, global_best_snap)
            if created_phase_logger and phase_logger is not None:
                phase_logger.close()
            return float(global_best_val), final_model
    else:
        best_val, best_snap, best_candidate_dir, best_epoch, final_epoch = train(model, search_phase=current_phase)
        model = restore(model, best_snap)
        global_best_val = float(best_val)
        global_best_snap = best_snap
        best_checkpoint = best_candidate_dir / "checkpoint_best.pt"
        improvements.append((total_neurons(model), best_val))
        record_phase_progress(
            {
                "module": module_name,
                "mode": mode,
                "candidate_index": int(candidate_index - 1),
                "candidate_dir": best_candidate_dir.name,
                "architecture": describe(model),
                "best_val": float(best_val),
                "best_epoch": int(best_epoch),
                "final_epoch": int(final_epoch),
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(best_candidate_dir / "checkpoint_last.pt"),
                "improved_over_global": True,
                "search_phase": current_phase,
                "width_fail": int(width_fail),
                "depth_fail": int(depth_fail),
            }
        )
        save_search_state(
            {
                "completed": False,
                "module": module_name,
                "mode": mode,
                "candidate_index": int(candidate_index),
                "current_phase": current_phase,
                "best_val": float(global_best_val),
                "best_candidate_dir": best_candidate_dir.name,
                "best_checkpoint": str(best_checkpoint),
                "width_fail": int(width_fail),
                "depth_fail": int(depth_fail),
                "width_stage_margin_fail": int(width_stage_margin_fail),
            }
        )

    def update_global_best(cur_model: Any, cand_val: float, cand_snap: Any, candidate_dir: Path, delta: float) -> bool:
        nonlocal global_best_val, global_best_snap, best_candidate_dir, best_checkpoint
        if cand_val < global_best_val - delta:
            global_best_val = cand_val
            global_best_snap = cand_snap
            best_candidate_dir = candidate_dir
            best_checkpoint = candidate_dir / "checkpoint_best.pt"
            improvements.append((total_neurons(cur_model), global_best_val))
            return True
        return False

    def persist_search_progress(
        *,
        phase: str,
        current_width_fail: int,
        current_depth_fail: int,
        current_margin_fail: int,
    ) -> None:
        save_search_state(
            {
                "completed": False,
                "module": module_name,
                "mode": mode,
                "candidate_index": int(candidate_index),
                "current_phase": phase,
                "best_val": float(global_best_val),
                "best_candidate_dir": best_candidate_dir.name if best_candidate_dir is not None else None,
                "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
                "width_fail": int(current_width_fail),
                "depth_fail": int(current_depth_fail),
                "width_stage_margin_fail": int(current_margin_fail),
            }
        )

    def ensure_uniform_width(
        cur_model: Any,
        *,
        update_global: bool = True,
        current_width_fail: int = 0,
        current_depth_fail: int = 0,
        current_margin_fail: int = 0,
    ) -> Tuple[Any, bool, Optional[float], Any, int, int, int]:
        progressed = False
        last_val: Optional[float] = None
        last_snap: Any = None
        while True:
            _, _, widths = snapshot_shape(cur_model)
            if _widths_are_uniform(widths):
                return cur_model, progressed, last_val, last_snap, current_width_fail, current_depth_fail, current_margin_fail
            candidate = _try_expand_once(module_globals, cur_model, acfg, device, "width", expand_width_fn)
            if candidate is None:
                return cur_model, progressed, last_val, last_snap, current_width_fail, current_depth_fail, current_margin_fail
            if phase_logger is not None:
                phase_logger.log_console(f"[STAGED][WIDTH-FILL] {describe(cur_model)} -> {describe(candidate)}")
            cand_val, cand_snap, candidate_dir, best_epoch, final_epoch = train(candidate, search_phase="width_fill")
            cur_model = restore(candidate, cand_snap)
            last_val = cand_val
            last_snap = cand_snap
            improved_global = False
            if update_global:
                improved_global = update_global_best(cur_model, cand_val, cand_snap, candidate_dir, delta_width)
                if improved_global:
                    persist_search_progress(
                        phase="width_fill",
                        current_width_fail=current_width_fail,
                        current_depth_fail=current_depth_fail,
                        current_margin_fail=current_margin_fail,
                    )
            progressed = True
            record_phase_progress(
                {
                    "module": module_name,
                    "mode": mode,
                    "candidate_index": int(candidate_index - 1),
                    "candidate_dir": candidate_dir.name,
                    "architecture": describe(cur_model),
                    "best_val": float(cand_val),
                    "best_epoch": int(best_epoch),
                    "final_epoch": int(final_epoch),
                    "best_checkpoint": str(candidate_dir / "checkpoint_best.pt"),
                    "last_checkpoint": str(candidate_dir / "checkpoint_last.pt"),
                    "improved_over_global": bool(improved_global),
                    "search_phase": "width_fill",
                    "width_fail": int(current_width_fail),
                    "depth_fail": int(current_depth_fail),
                }
            )

    def run_width_stage(cur_model: Any, *, current_width_fail: int, current_depth_fail: int, current_margin_fail: int) -> Tuple[Any, bool, bool, float, int, int, int]:
        stage_anchor = float(global_best_val)
        progressed = False
        while True:
            candidate = _try_expand_once(module_globals, cur_model, acfg, device, "width", expand_width_fn)
            if candidate is None:
                return cur_model, False, False, 0.0, current_width_fail, current_depth_fail, current_margin_fail
            if phase_logger is not None:
                phase_logger.log_console(f"[STAGED][WIDTH] {describe(cur_model)} -> {describe(candidate)}")
            cand_val, cand_snap, candidate_dir, best_epoch, final_epoch = train(candidate, search_phase="width")
            cur_model = restore(candidate, cand_snap)
            improved_global = update_global_best(cur_model, cand_val, cand_snap, candidate_dir, delta_width)
            if improved_global:
                persist_search_progress(
                    phase="width",
                    current_width_fail=current_width_fail,
                    current_depth_fail=current_depth_fail,
                    current_margin_fail=current_margin_fail,
                )
            progressed = True
            record_phase_progress(
                {
                    "module": module_name,
                    "mode": mode,
                    "candidate_index": int(candidate_index - 1),
                    "candidate_dir": candidate_dir.name,
                    "architecture": describe(cur_model),
                    "best_val": float(cand_val),
                    "best_epoch": int(best_epoch),
                    "final_epoch": int(final_epoch),
                    "best_checkpoint": str(candidate_dir / "checkpoint_best.pt"),
                    "last_checkpoint": str(candidate_dir / "checkpoint_last.pt"),
                    "improved_over_global": bool(improved_global),
                    "search_phase": "width",
                    "width_fail": int(current_width_fail),
                    "depth_fail": int(current_depth_fail),
                }
            )
            _, _, widths = snapshot_shape(cur_model)
            if _widths_are_uniform(widths):
                stage_pct = _pct_improvement(stage_anchor, global_best_val)
                return cur_model, progressed, (global_best_val < stage_anchor - delta_width), stage_pct, current_width_fail, current_depth_fail, current_margin_fail

    def run_width_phase(cur_model: Any, *, initial_width_fail: int = 0, initial_margin_fail: int = 0, current_depth_fail: int = 0) -> Tuple[Any, bool, int, int]:
        local_width_fail = int(initial_width_fail)
        local_margin_fail = int(initial_margin_fail)
        any_phase_improvement = False
        while True:
            before_val = float(global_best_val)
            cur_model, completed, stage_improved, stage_pct, _, _, _ = run_width_stage(
                cur_model,
                current_width_fail=local_width_fail,
                current_depth_fail=current_depth_fail,
                current_margin_fail=local_margin_fail,
            )
            if not completed:
                break
            any_phase_improvement = any_phase_improvement or stage_improved
            local_width_fail = 0 if stage_improved else local_width_fail + 1
            local_margin_fail = 0 if stage_pct >= width_stage_min_improve_pct else local_margin_fail + 1
            if phase_logger is not None:
                phase_logger.log_console(
                    f"[STAGED][WIDTH] completed arch={describe(cur_model)} "
                    f"stage_improved={stage_improved} stage_pct={stage_pct:.4f} "
                    f"global_best={global_best_val:.6f} width_fail={local_width_fail}/{patience_width} "
                    f"margin_fail={local_margin_fail}/{width_stage_margin_patience}"
                )
            persist_search_progress(
                phase="width",
                current_width_fail=local_width_fail,
                current_depth_fail=current_depth_fail,
                current_margin_fail=local_margin_fail,
            )
            if local_width_fail >= patience_width or local_margin_fail >= width_stage_margin_patience:
                break
            if float(global_best_val) >= before_val and not stage_improved and local_margin_fail >= width_stage_margin_patience:
                break
        cur_model, _, _, _, _, _, _ = ensure_uniform_width(
            cur_model,
            current_width_fail=local_width_fail,
            current_depth_fail=current_depth_fail,
            current_margin_fail=local_margin_fail,
        )
        return cur_model, any_phase_improvement, local_width_fail, local_margin_fail

    def run_depth_step(cur_model: Any, *, compare_after_warmup: bool = True, current_width_fail: int = 0, current_depth_fail: int = 0, current_margin_fail: int = 0) -> Tuple[Any, bool, bool, int, int, int]:
        cur_model, _, _, _, current_width_fail, current_depth_fail, current_margin_fail = ensure_uniform_width(
            cur_model,
            current_width_fail=current_width_fail,
            current_depth_fail=current_depth_fail,
            current_margin_fail=current_margin_fail,
        )
        _, _, widths = snapshot_shape(cur_model)
        if not _can_spawn_new_depth_layer(widths, min_new_layer_width):
            return cur_model, False, False, current_width_fail, current_depth_fail, current_margin_fail
        candidate = _try_expand_once(module_globals, cur_model, acfg, device, "depth", expand_depth_fn)
        if candidate is None:
            return cur_model, False, False, current_width_fail, current_depth_fail, current_margin_fail
        if phase_logger is not None:
            phase_logger.log_console(f"[STAGED][DEPTH] {describe(cur_model)} -> {describe(candidate)}")
        _, _, candidate_widths = snapshot_shape(candidate)
        if _widths_are_uniform(candidate_widths):
            cand_val, cand_snap, candidate_dir, best_epoch, final_epoch = train(candidate, search_phase="depth")
            cur_model = restore(candidate, cand_snap)
            improved = False
            if compare_after_warmup:
                improved = update_global_best(cur_model, cand_val, cand_snap, candidate_dir, delta_depth)
                if improved:
                    persist_search_progress(
                        phase="depth",
                        current_width_fail=current_width_fail,
                        current_depth_fail=current_depth_fail,
                        current_margin_fail=current_margin_fail,
                    )
            record_phase_progress(
                {
                    "module": module_name,
                    "mode": mode,
                    "candidate_index": int(candidate_index - 1),
                    "candidate_dir": candidate_dir.name,
                    "architecture": describe(cur_model),
                    "best_val": float(cand_val),
                    "best_epoch": int(best_epoch),
                    "final_epoch": int(final_epoch),
                    "best_checkpoint": str(candidate_dir / "checkpoint_best.pt"),
                    "last_checkpoint": str(candidate_dir / "checkpoint_last.pt"),
                    "improved_over_global": bool(improved),
                    "search_phase": "depth",
                    "width_fail": int(current_width_fail),
                    "depth_fail": int(current_depth_fail),
                }
            )
            return cur_model, True, improved, current_width_fail, current_depth_fail, current_margin_fail
        warmup_val, warmup_snap, warmup_dir, warmup_best_epoch, warmup_final_epoch = train(candidate, search_phase="depth_warmup")
        warm_model = restore(candidate, warmup_snap)
        warmup_improved = False
        if compare_after_warmup:
            warmup_improved = update_global_best(warm_model, warmup_val, warmup_snap, warmup_dir, delta_depth)
            if warmup_improved:
                persist_search_progress(
                    phase="depth_warmup",
                    current_width_fail=current_width_fail,
                    current_depth_fail=current_depth_fail,
                    current_margin_fail=current_margin_fail,
                )
        record_phase_progress(
            {
                "module": module_name,
                "mode": mode,
                "candidate_index": int(candidate_index - 1),
                "candidate_dir": warmup_dir.name,
                "architecture": describe(warm_model),
                "best_val": float(warmup_val),
                "best_epoch": int(warmup_best_epoch),
                "final_epoch": int(warmup_final_epoch),
                "best_checkpoint": str(warmup_dir / "checkpoint_best.pt"),
                "last_checkpoint": str(warmup_dir / "checkpoint_last.pt"),
                "improved_over_global": bool(warmup_improved),
                "search_phase": "depth_warmup",
                "width_fail": int(current_width_fail),
                "depth_fail": int(current_depth_fail),
            }
        )
        depth_anchor = float(global_best_val)
        warmed_model, _, warmed_val, warmed_snap, current_width_fail, current_depth_fail, current_margin_fail = ensure_uniform_width(
            warm_model,
            update_global=True,
            current_width_fail=current_width_fail,
            current_depth_fail=current_depth_fail,
            current_margin_fail=current_margin_fail,
        )
        if warmed_val is None or warmed_snap is None:
            cur_model = restore(warm_model, warmup_snap)
            return cur_model, True, False, current_width_fail, current_depth_fail, current_margin_fail
        cur_model = restore(warmed_model, warmed_snap)
        improved = bool(compare_after_warmup and float(global_best_val) < depth_anchor - delta_depth)
        return cur_model, True, improved, current_width_fail, current_depth_fail, current_margin_fail

    def run_depth_phase(cur_model: Any, *, initial_depth_fail: int = 0, current_width_fail: int = 0, current_margin_fail: int = 0) -> Tuple[Any, bool, int]:
        cur_model, _, _, _, current_width_fail, initial_depth_fail, current_margin_fail = ensure_uniform_width(
            cur_model,
            current_width_fail=current_width_fail,
            current_depth_fail=initial_depth_fail,
            current_margin_fail=current_margin_fail,
        )
        local_depth_fail = int(initial_depth_fail)
        depth_margin_fail = 0
        depth_stage_anchor = float(global_best_val)
        any_phase_improvement = False
        while local_depth_fail < patience_depth and depth_margin_fail < depth_stage_margin_patience:
            before_val = float(global_best_val)
            cur_model, progressed, improved, _, _, _ = run_depth_step(
                cur_model,
                current_width_fail=current_width_fail,
                current_depth_fail=local_depth_fail,
                current_margin_fail=current_margin_fail,
            )
            stage_pct = _pct_improvement(depth_stage_anchor, global_best_val)
            if phase_logger is not None:
                phase_logger.log_console(
                    f"[STAGED][DEPTH] arch={describe(cur_model)} progressed={progressed} improved={improved} "
                    f"global_best={global_best_val:.6f} depth_fail={local_depth_fail if improved else local_depth_fail + 1}/{patience_depth} "
                    f"depth_margin_fail={depth_margin_fail if stage_pct >= depth_stage_min_improve_pct else depth_margin_fail + 1}/{depth_stage_margin_patience} "
                    f"stage_pct={stage_pct:.4f}"
                )
            if not progressed:
                break
            if not improved:
                probe_snap = _snapshot(module_globals, cur_model)
                _, _, probe_widths = snapshot_shape(cur_model)
                if not _can_spawn_new_depth_layer(probe_widths, min_new_layer_width):
                    break
                if _try_expand_once(module_globals, cur_model, acfg, device, "depth", expand_depth_fn) is None:
                    break
                cur_model = restore(cur_model, probe_snap)
            if not improved:
                local_depth_fail += 1
            else:
                local_depth_fail = 0
                any_phase_improvement = True
            depth_margin_fail = 0 if stage_pct >= depth_stage_min_improve_pct else depth_margin_fail + 1
            depth_stage_anchor = float(global_best_val)
            persist_search_progress(
                phase="depth",
                current_width_fail=current_width_fail,
                current_depth_fail=local_depth_fail,
                current_margin_fail=current_margin_fail,
            )
            if float(global_best_val) >= before_val and not improved and depth_margin_fail >= depth_stage_margin_patience:
                break
        return cur_model, any_phase_improvement, local_depth_fail

    if mode in ("width_only", "width"):
        model, _, width_fail, width_stage_margin_fail = run_width_phase(
            model,
            initial_width_fail=width_fail,
            initial_margin_fail=width_stage_margin_fail,
            current_depth_fail=depth_fail,
        )
    elif mode in ("depth_only", "depth"):
        model, _, depth_fail = run_depth_phase(model, initial_depth_fail=depth_fail, current_width_fail=width_fail)
    elif mode == "width_to_depth":
        while depth_fail < patience_depth:
            model, _, width_fail, width_stage_margin_fail = run_width_phase(
                model,
                initial_width_fail=width_fail,
                initial_margin_fail=width_stage_margin_fail,
                current_depth_fail=depth_fail,
            )
            current_phase = "depth"
            model, progressed, improved, width_fail, depth_fail, width_stage_margin_fail = run_depth_step(
                model,
                compare_after_warmup=True,
                current_width_fail=width_fail,
                current_depth_fail=depth_fail,
                current_margin_fail=width_stage_margin_fail,
            )
            if not progressed:
                break
            if improved:
                depth_fail = 0
            else:
                depth_fail += 1
            width_fail = 0
            width_stage_margin_fail = 0
            current_phase = "width"
    elif mode == "depth_to_width":
        while width_fail < patience_width:
            model, _, depth_fail = run_depth_phase(model, initial_depth_fail=depth_fail, current_width_fail=width_fail)
            before_cycle = float(global_best_val)
            model, width_improved, width_fail, width_stage_margin_fail = run_width_phase(
                model,
                initial_width_fail=width_fail,
                initial_margin_fail=width_stage_margin_fail,
                current_depth_fail=depth_fail,
            )
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
                model, improved, width_fail, width_stage_margin_fail = run_width_phase(
                    model,
                    initial_width_fail=width_fail,
                    initial_margin_fail=width_stage_margin_fail,
                    current_depth_fail=depth_fail,
                )
                width_done = not (improved or float(global_best_val) < before - delta_width)
                phase = "depth"
            else:
                before = float(global_best_val)
                model, improved, depth_fail = run_depth_phase(model, initial_depth_fail=depth_fail, current_width_fail=width_fail)
                depth_done = not (improved or float(global_best_val) < before - delta_depth)
                phase = "width"
    elif mode == "alt_depth":
        width_done = False
        depth_done = False
        phase = "depth"
        while not (width_done and depth_done):
            if phase == "depth":
                before = float(global_best_val)
                model, improved, depth_fail = run_depth_phase(model, initial_depth_fail=depth_fail, current_width_fail=width_fail)
                depth_done = not (improved or float(global_best_val) < before - delta_depth)
                phase = "width"
            else:
                before = float(global_best_val)
                model, improved, width_fail, width_stage_margin_fail = run_width_phase(
                    model,
                    initial_width_fail=width_fail,
                    initial_margin_fail=width_stage_margin_fail,
                    current_depth_fail=depth_fail,
                )
                width_done = not (improved or float(global_best_val) < before - delta_width)
                phase = "depth"
    else:
        raise ValueError(f"Unsupported ADP mode: {mode}")

    model = restore(model, global_best_snap)
    save_search_state(
        {
            "completed": True,
            "module": module_name,
            "mode": mode,
            "candidate_index": int(candidate_index),
            "current_phase": current_phase,
            "best_val": float(global_best_val),
            "best_candidate_dir": best_candidate_dir.name if best_candidate_dir is not None else None,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
            "width_fail": int(width_fail),
            "depth_fail": int(depth_fail),
            "width_stage_margin_fail": int(width_stage_margin_fail),
        }
    )
    _write_json(
        summary_path,
        {
            "module": module_name,
            "mode": mode,
            "best_val": float(global_best_val),
            "best_candidate_dir": best_candidate_dir.name if best_candidate_dir is not None else None,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
            "architecture": describe(model),
        },
    )
    if log_loss and plot_loss_vs_epoch is not None:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{module_globals.get('__name__', 'adp')} ({mode})")
    if log_neurons and improvements and plot_loss_vs_neurons is not None:
        plot_loss_vs_neurons([n for n, _ in improvements], [v for _, v in improvements], results_dir / "loss_vs_neurons.png", title=f"{module_globals.get('__name__', 'adp')} ({mode})")
    if created_phase_logger and phase_logger is not None:
        phase_logger.close()
    return float(global_best_val), model
