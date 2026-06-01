from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


SUPPORTED_MODES = (
    "width_only",
    "depth_only",
    "alt_width",
    "alt_depth",
    "width_to_depth",
    "depth_to_width",
)

_FFN_ATTRIBUTE_NAMES = {"mlp", "ffn", "feed_forward", "feedforward"}


def _new_linear_like(source: nn.Linear, in_features: int, out_features: int) -> nn.Linear:
    target = nn.Linear(
        int(in_features),
        int(out_features),
        bias=source.bias is not None,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        rows = min(target.weight.size(0), source.weight.size(0))
        cols = min(target.weight.size(1), source.weight.size(1))
        target.weight[:rows, :cols].copy_(source.weight[:rows, :cols])
        if target.bias is not None and source.bias is not None:
            target.bias[:rows].copy_(source.bias[:rows])
    return target


def _activation_copy(module: Any) -> nn.Module:
    if isinstance(module, nn.Module):
        return copy.deepcopy(module)
    if callable(module):
        return _CallableActivation(module)
    return nn.GELU()


class _CallableActivation(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class ExpandableFFN(nn.Module):
    """A transformer FFN whose internal MLP shape can grow without touching attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_widths: Sequence[int],
        output_dim: int,
        activation: nn.Module,
        dropout: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if not hidden_widths:
            raise ValueError("ExpandableFFN requires at least one hidden width")
        self.input_dim = int(input_dim)
        self.hidden_widths = [int(width) for width in hidden_widths]
        self.output_dim = int(output_dim)
        self.activation = copy.deepcopy(activation)
        self.dropout = copy.deepcopy(dropout) if dropout is not None else nn.Identity()
        self.hidden_layers = nn.ModuleList()
        previous = self.input_dim
        for width in self.hidden_widths:
            self.hidden_layers.append(nn.Linear(previous, width))
            previous = width
        self.output = nn.Linear(previous, self.output_dim)

    @classmethod
    def from_linears(
        cls,
        first: nn.Linear,
        last: nn.Linear,
        activation: nn.Module,
        dropout: Optional[nn.Module] = None,
    ) -> "ExpandableFFN":
        module = cls(first.in_features, [first.out_features], last.out_features, activation, dropout)
        module.hidden_layers[0] = _new_linear_like(first, first.in_features, first.out_features)
        module.output = _new_linear_like(last, last.in_features, last.out_features)
        return module

    def widen_staged(self, steps: int, max_width: int) -> bool:
        changed = False
        for _ in range(max(1, int(steps))):
            eligible = [
                index for index, width in enumerate(self.hidden_widths)
                if int(width) < int(max_width)
            ]
            if not eligible:
                break
            minimum = min(self.hidden_widths[index] for index in eligible)
            target_index = next(index for index in eligible if self.hidden_widths[index] == minimum)
            self.hidden_widths[target_index] += 1
            changed = True
        if changed:
            self._rebuild_linears()
        return changed

    def deepen(self, max_depth: int, min_new_layer_width: int) -> bool:
        if len(self.hidden_widths) >= int(max_depth) or not self.is_uniform:
            return False
        if int(self.hidden_widths[-1]) <= int(min_new_layer_width):
            return False
        self.hidden_widths.append(int(min_new_layer_width))
        self._rebuild_linears()
        return True

    @property
    def is_uniform(self) -> bool:
        return bool(self.hidden_widths) and len(set(self.hidden_widths)) == 1

    def _rebuild_linears(self) -> None:
        old_hidden = list(self.hidden_layers)
        old_output = self.output
        rebuilt = nn.ModuleList()
        previous = self.input_dim
        for index, width in enumerate(self.hidden_widths):
            if index < len(old_hidden):
                rebuilt.append(_new_linear_like(old_hidden[index], previous, width))
            else:
                template = old_hidden[-1]
                rebuilt.append(_new_linear_like(template, previous, width))
            previous = width
        self.hidden_layers = rebuilt
        self.output = _new_linear_like(old_output, previous, self.output_dim)

    def forward(self, x: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = self.dropout(self.activation(layer(x)))
        return self.dropout(self.output(x))


def _simple_linears(module: nn.Module) -> Optional[Tuple[nn.Linear, nn.Linear, nn.Module, Optional[nn.Module]]]:
    if isinstance(module, ExpandableFFN):
        return None
    if isinstance(module, nn.Sequential):
        linears = [child for child in module if isinstance(child, nn.Linear)]
        if len(linears) == 2:
            activation = next(
                (child for child in module if isinstance(child, (nn.ReLU, nn.GELU, nn.SiLU, nn.Hardswish))),
                nn.GELU(),
            )
            dropout = next((child for child in module if isinstance(child, nn.Dropout)), None)
            return linears[0], linears[1], activation, dropout
    first = getattr(module, "fc1", None)
    last = getattr(module, "fc2", None)
    if isinstance(first, nn.Linear) and isinstance(last, nn.Linear):
        activation = getattr(module, "act", getattr(module, "a", nn.GELU()))
        dropout = getattr(module, "drop", None)
        return first, last, _activation_copy(activation), dropout if isinstance(dropout, nn.Module) else None
    return None


def normalize_transformer_ffns(model: nn.Module) -> List[ExpandableFFN]:
    """Convert recognized transformer FFNs to one synchronized, expandable representation."""
    converted: List[ExpandableFFN] = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, ExpandableFFN):
                converted.append(child)
                continue
            is_named_ffn = name in _FFN_ATTRIBUTE_NAMES
            is_typed_ffn = child.__class__.__name__.lower() in {
                "mlp",
                "ffn",
                "feedforward",
                "feedforwardmodule",
            }
            if not is_named_ffn and not is_typed_ffn:
                continue
            parts = _simple_linears(child)
            if parts is None:
                continue
            first, last, activation, dropout = parts
            replacement = ExpandableFFN.from_linears(first, last, activation, dropout)
            setattr(parent, name, replacement)
            converted.append(replacement)

    for layer in model.modules():
        if not isinstance(layer, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            continue
        marker = getattr(layer, "_adp_expandable_ffn", None)
        if isinstance(marker, ExpandableFFN):
            converted.append(marker)
            continue
        replacement = ExpandableFFN.from_linears(
            layer.linear1,
            layer.linear2,
            _activation_copy(layer.activation),
            layer.dropout,
        )
        layer.linear1 = replacement.hidden_layers[0]
        layer.linear2 = replacement.output
        layer.activation = _TransformerLayerActivation(replacement)
        layer._adp_expandable_ffn = replacement
        converted.append(replacement)

    unique: List[ExpandableFFN] = []
    seen = set()
    for ffn in converted:
        if id(ffn) not in seen:
            unique.append(ffn)
            seen.add(id(ffn))
    if not unique:
        raise ValueError("No supported transformer FFN/MLP blocks were found")
    return unique


class _TransformerLayerActivation(nn.Module):
    """Runs the first activation plus any ADP-added hidden FFN layers."""

    def __init__(self, ffn: ExpandableFFN) -> None:
        super().__init__()
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn.activation(x)
        for layer in list(self.ffn.hidden_layers)[1:]:
            x = self.ffn.dropout(self.ffn.activation(layer(x)))
        return x


def _refresh_native_transformer_linears(model: nn.Module) -> None:
    for layer in model.modules():
        marker = getattr(layer, "_adp_expandable_ffn", None)
        if isinstance(marker, ExpandableFFN):
            layer.linear1 = marker.hidden_layers[0]
            layer.linear2 = marker.output
            layer.activation = _TransformerLayerActivation(marker)


def transformer_ffn_architectures(model: nn.Module) -> List[List[int]]:
    return [list(ffn.hidden_widths) for ffn in normalize_transformer_ffns(model)]


def transformer_ffn_neurons(model: nn.Module) -> int:
    return sum(sum(widths) for widths in transformer_ffn_architectures(model))


def transformer_ffns_are_uniform(model: nn.Module) -> bool:
    return all(ffn.is_uniform for ffn in normalize_transformer_ffns(model))


def expand_transformer_ffn_width(
    model: nn.Module,
    ex_k: int = 1,
    max_width: int = 4096,
    max_neurons: Optional[int] = None,
) -> Optional[nn.Module]:
    candidate = copy.deepcopy(model)
    ffns = normalize_transformer_ffns(candidate)
    changed = [ffn.widen_staged(ex_k, max_width) for ffn in ffns]
    if not all(changed):
        return None
    _refresh_native_transformer_linears(candidate)
    if max_neurons is not None and transformer_ffn_neurons(candidate) > int(max_neurons):
        return None
    return candidate


def expand_transformer_ffn_depth(
    model: nn.Module,
    max_depth: int = 16,
    min_new_layer_width: int = 10,
    max_neurons: Optional[int] = None,
) -> Optional[nn.Module]:
    candidate = copy.deepcopy(model)
    ffns = normalize_transformer_ffns(candidate)
    changed = [ffn.deepen(max_depth, min_new_layer_width) for ffn in ffns]
    if not all(changed):
        return None
    _refresh_native_transformer_linears(candidate)
    if max_neurons is not None and transformer_ffn_neurons(candidate) > int(max_neurons):
        return None
    return candidate


@dataclass
class SearchConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    width_expansion_patience: int = 10
    depth_expansion_patience: int = 2
    width_stage_margin_patience: int = 10
    width_stage_min_improve_pct: float = 1.0
    ex_k_width: int = 1
    max_width: int = 4096
    max_depth: int = 16
    max_neurons: int = 5_000_000
    min_new_layer_width: int = 10


@dataclass
class SearchState:
    mode: str
    current_phase: str
    best_val: float = float("inf")
    width_fail: int = 0
    depth_fail: int = 0
    consecutive_fail: int = 0
    width_stage_improved: bool = False
    width_stage_margin_fail: int = 0
    width_stage_anchor_val: Optional[float] = None
    warmup_to_uniform: bool = False
    candidate_index: int = 0
    completed: bool = False


class StagedTransformerFFNSearch:
    """Current staged-width MLP ADP ordering applied to synchronized transformer FFNs."""

    def __init__(self, model: nn.Module, config: SearchConfig, state: Optional[SearchState] = None) -> None:
        if config.adp_mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported ADP mode: {config.adp_mode}")
        self.config = config
        self.model = copy.deepcopy(model)
        normalize_transformer_ffns(self.model)
        initial_phase = "width" if config.adp_mode in {"width_only", "alt_width", "width_to_depth"} else "depth"
        self.state = state or SearchState(mode=config.adp_mode, current_phase=initial_phase)
        self.best_model: Optional[nn.Module] = None

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self.state)

    def next_candidate(self) -> Optional[Tuple[str, nn.Module]]:
        if self.state.completed:
            return None
        if self.state.candidate_index == 0:
            return self.state.current_phase, copy.deepcopy(self.model)
        phase = self._select_phase()
        if phase is None:
            self.state.completed = True
            return None
        attempted = set()
        for _ in range(4):
            if phase in attempted:
                self.state.completed = True
                return None
            attempted.add(phase)
            candidate = (
                expand_transformer_ffn_width(
                    self.model,
                    self.config.ex_k_width,
                    self.config.max_width,
                    self.config.max_neurons,
                )
                if phase == "width"
                else expand_transformer_ffn_depth(
                    self.model,
                    self.config.max_depth,
                    self.config.min_new_layer_width,
                    self.config.max_neurons,
                )
            )
            if candidate is not None:
                return phase, candidate
            phase = self._fallback_phase_after_exhaustion(phase)
            if phase is None:
                self.state.completed = True
                return None
        raise RuntimeError("ADP phase fallback did not converge")

    def record_result(self, phase: str, candidate: nn.Module, best_val: float) -> bool:
        if phase not in {"width", "depth"}:
            raise ValueError(f"Unknown search phase: {phase}")
        improved = float(best_val) < (float(self.state.best_val) - float(self.config.delta))
        if improved:
            self.state.best_val = float(best_val)
            self.best_model = copy.deepcopy(candidate)
        self.model = copy.deepcopy(candidate)
        uniform = transformer_ffns_are_uniform(candidate)
        if phase == "width":
            self._record_width_result(improved, uniform)
        else:
            self._record_depth_result(improved, uniform)
        self.state.candidate_index += 1
        self._advance_after_result(phase, uniform)
        return improved

    def _record_width_result(self, improved: bool, uniform: bool) -> None:
        if self.state.warmup_to_uniform:
            if uniform:
                self.state.warmup_to_uniform = False
                self.state.width_fail = 0
                self.state.width_stage_margin_fail = 0
                self.state.width_stage_improved = False
                self.state.width_stage_anchor_val = float(self.state.best_val)
            return
        self.state.width_stage_improved = self.state.width_stage_improved or improved
        if not uniform:
            return
        margin = _relative_improvement_pct(self.state.width_stage_anchor_val, self.state.best_val)
        if self.state.width_stage_improved:
            self.state.width_fail = 0
            self.state.consecutive_fail = 0
        else:
            self.state.width_fail += 1
            self.state.consecutive_fail += 1
        if self.state.width_stage_anchor_val is None or margin >= self.config.width_stage_min_improve_pct:
            self.state.width_stage_margin_fail = 0
        else:
            self.state.width_stage_margin_fail += 1
        self.state.width_stage_anchor_val = float(self.state.best_val)
        self.state.width_stage_improved = False

    def _record_depth_result(self, improved: bool, uniform: bool) -> None:
        if not uniform:
            self.state.depth_fail = 0
            self.state.warmup_to_uniform = True
            self.state.width_fail = 0
            self.state.width_stage_margin_fail = 0
            self.state.width_stage_improved = False
            self.state.width_stage_anchor_val = float(self.state.best_val)
        elif improved:
            self.state.depth_fail = 0
            self.state.consecutive_fail = 0
        else:
            self.state.depth_fail += 1
            self.state.consecutive_fail += 1

    def _select_phase(self) -> Optional[str]:
        mode = self.state.mode
        phase = self.state.current_phase
        uniform = transformer_ffns_are_uniform(self.model)
        if phase == "depth" and not uniform and mode in {"alt_width", "alt_depth", "width_to_depth", "depth_to_width"}:
            self.state.current_phase = "width"
            self.state.depth_fail = 0
            self.state.warmup_to_uniform = True
            return "width"
        if mode == "width_only":
            return None if self._width_limit_hit() else "width"
        if mode == "depth_only":
            return None if self.state.depth_fail >= self.config.depth_expansion_patience else "depth"
        if mode == "width_to_depth":
            if phase == "width" and uniform and self._width_limit_hit():
                self.state.current_phase = "depth"
                return "depth"
            if phase == "depth" and self.state.depth_fail >= self.config.depth_expansion_patience:
                return None
            return phase
        if mode == "depth_to_width":
            if phase == "depth" and self.state.depth_fail >= self.config.depth_expansion_patience:
                self.state.current_phase = "width"
                return "width"
            if phase == "width" and uniform and self._width_limit_hit():
                return None
            return phase
        if mode == "alt_width":
            if phase == "width" and uniform and self._width_limit_hit():
                self.state.current_phase = "depth"
                return "depth"
            if phase == "depth" and self.state.depth_fail >= self.config.depth_expansion_patience:
                self.state.current_phase = "width"
                self.state.depth_fail = 0
                return "width"
            return phase
        if phase == "depth" and self.state.depth_fail >= self.config.depth_expansion_patience:
            self.state.current_phase = "width"
            self.state.depth_fail = 0
            return "width"
        if phase == "width" and uniform and self._width_limit_hit():
            self.state.current_phase = "depth"
            self.state.width_fail = 0
            self.state.width_stage_margin_fail = 0
            return "depth"
        return phase

    def _fallback_phase_after_exhaustion(self, phase: str) -> Optional[str]:
        mode = self.state.mode
        if phase == "width":
            if mode in {"alt_width", "alt_depth", "width_to_depth"} and transformer_ffns_are_uniform(self.model):
                self.state.current_phase = "depth"
                return "depth"
            return None
        if mode in {"alt_width", "alt_depth", "width_to_depth", "depth_to_width"}:
            self.state.current_phase = "width"
            self.state.depth_fail = 0
            return "width"
        return None

    def _advance_after_result(self, phase: str, uniform: bool) -> None:
        mode = self.state.mode
        if mode == "width_to_depth" and phase == "depth":
            self.state.current_phase = "width"
            self.state.width_fail = 0
            self.state.width_stage_margin_fail = 0
        elif mode == "depth_to_width" and phase == "width":
            self.state.current_phase = "depth" if uniform else "width"
            if uniform:
                self.state.depth_fail = 0

    def _width_limit_hit(self) -> bool:
        margin_hit = (
            self.config.width_stage_margin_patience > 0
            and self.state.width_stage_margin_fail >= self.config.width_stage_margin_patience
        )
        return self.state.width_fail >= self.config.width_expansion_patience or margin_hit


def _relative_improvement_pct(previous: Optional[float], current: float) -> float:
    if previous is None:
        return float("inf")
    return ((float(previous) - float(current)) / max(abs(float(previous)), 1e-12)) * 100.0


def run_staged_transformer_ffn_search(
    model: nn.Module,
    train_candidate: Callable[[nn.Module], float],
    config: SearchConfig,
) -> Tuple[float, nn.Module, StagedTransformerFFNSearch]:
    search = StagedTransformerFFNSearch(model, config)
    while True:
        item = search.next_candidate()
        if item is None:
            break
        phase, candidate = item
        search.record_result(phase, candidate, float(train_candidate(candidate)))
    final_model = search.best_model if search.best_model is not None else search.model
    return float(search.state.best_val), copy.deepcopy(final_model), search


def infer_top_level_model_class(baseline_module: Any) -> Optional[type[nn.Module]]:
    """Return the final model class declared by a baseline module, not an early helper block."""
    candidates = [
        value
        for value in vars(baseline_module).values()
        if isinstance(value, type)
        and issubclass(value, nn.Module)
        and value is not nn.Module
        and getattr(value, "__module__", None) == getattr(baseline_module, "__name__", None)
    ]
    return candidates[-1] if candidates else None


def bind_legacy_wrapper(module_globals: Dict[str, Any]) -> None:
    """Make an existing transformer wrapper delegate its FFN growth to this module."""
    baseline_module = module_globals.get("baseline_module")
    if baseline_module is not None:
        inferred_class = infer_top_level_model_class(baseline_module)
        if inferred_class is not None:
            module_globals["ModelClass"] = inferred_class

    def expand_width(model: nn.Module, ex_k: int = 1, max_width: int = 4096, device: Any = None, cfg: Any = None) -> Optional[nn.Module]:
        del device
        return expand_transformer_ffn_width(
            model,
            getattr(cfg, "ex_k_width", getattr(cfg, "ex_k", ex_k)) if cfg is not None else ex_k,
            max_width,
            getattr(cfg, "max_neurons", None) if cfg is not None else None,
        )

    def expand_depth(model: nn.Module, max_depth: int = 16, device: Any = None, cfg: Any = None) -> Optional[nn.Module]:
        del device, cfg
        return expand_transformer_ffn_depth(
            model,
            max_depth,
            getattr(cfg, "min_new_layer_width", 10) if cfg is not None else 10,
            getattr(cfg, "max_neurons", None) if cfg is not None else None,
        )

    def snapshot_arch_and_state(model: nn.Module, state_dict: Any = None) -> Dict[str, Any]:
        del state_dict
        return {"model": copy.deepcopy(model), "architectures": transformer_ffn_architectures(model)}

    def restore_arch_and_state(model: nn.Module, snap: Dict[str, Any], device: Any = None) -> nn.Module:
        del model
        restored = copy.deepcopy(snap["model"])
        return restored.to(device) if device is not None else restored

    def total_neurons(model: nn.Module, *_: Any, **__: Any) -> int:
        return transformer_ffn_neurons(model)

    def adp_search(
        model: nn.Module,
        dl_train: Iterable,
        dl_val: Iterable,
        acfg: Any,
        device: Any,
        log_loss: bool = False,
        log_neurons: bool = False,
        results_dir: Any = None,
    ) -> Tuple[float, nn.Module, int, int]:
        from utils.adp_contract import run_module_adp

        value, final_model = run_module_adp(
            module_globals,
            model,
            dl_train,
            dl_val,
            acfg,
            device,
            log_loss=log_loss,
            log_neurons=log_neurons,
            results_dir=results_dir,
        )
        architectures = transformer_ffn_architectures(final_model)
        canonical = architectures[0]
        return value, final_model.to(device), max(canonical), len(canonical)

    module_globals.update(
        {
            "expand_width": expand_width,
            "expand_depth": expand_depth,
            "snapshot_arch_and_state": snapshot_arch_and_state,
            "restore_arch_and_state": restore_arch_and_state,
            "total_neurons": total_neurons,
            "adp_search": adp_search,
        }
    )
