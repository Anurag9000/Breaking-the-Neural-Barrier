from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt


def _resize_linear(old: nn.Linear, new_out: int, new_in: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        if old.bias is not None and new.bias is not None:
            new.bias[:r] = old.bias[:r]
    return new


def _copy_sequence_linears(old_linears: Sequence[nn.Linear], shapes: Iterable[Tuple[int, int]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for idx, (new_in, new_out) in enumerate(shapes):
        old = old_linears[idx] if idx < len(old_linears) else None
        if old is None:
            linear = nn.Linear(new_in, new_out)
        else:
            linear = _resize_linear(old, new_out, new_in)
        layers.append(linear)
        if idx != len(list(shapes)) - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def infer_hidden_widths(model: Any) -> List[int]:
    widths = getattr(model, "hidden_widths", None)
    if widths is not None:
        return [int(w) for w in widths]
    dae = getattr(model, "dae", None)
    dae_widths = getattr(dae, "hidden_widths", None)
    if dae_widths is not None:
        return [int(w) for w in dae_widths]
    width = int(getattr(model, "width", getattr(dae, "width", 0)))
    depth = int(getattr(model, "depth", getattr(dae, "depth", 0)))
    return [width for _ in range(max(0, depth))]


def next_staged_widths(hidden_widths: Sequence[int], max_width: int, ex_k: int) -> List[int]:
    widths = [int(w) for w in hidden_widths]
    if not widths:
        return widths
    if len(set(widths)) == 1:
        target = min(int(max_width), max(widths) + max(1, int(ex_k)))
    else:
        target = max(widths)
    next_widths = list(widths)
    for idx, width in enumerate(next_widths):
        if width < target:
            next_widths[idx] = width + 1
            break
    return next_widths


def rebuild_unsup_dae_model(model_cls: Type[Any], model: Any, hidden_widths: Sequence[int], device: torch.device) -> Any:
    widths = [int(w) for w in hidden_widths]
    if not widths:
        raise ValueError("hidden_widths must be non-empty")
    new_model = model_cls(
        in_channels=model.in_channels,
        img_size=model.img_size,
        width=max(widths),
        depth=len(widths),
    ).to(device)

    old_state = copy.deepcopy(model.state_dict())
    old_enc = [m for m in model.encoder if isinstance(m, nn.Linear)]
    old_dec = [m for m in model.decoder if isinstance(m, nn.Linear)]

    enc_shapes = []
    prev = int(model.input_dim)
    for width in widths:
        enc_shapes.append((prev, int(width)))
        prev = int(width)
    dec_shapes = []
    rev_widths = list(reversed(widths))
    prev = rev_widths[0]
    for width in rev_widths[1:]:
        dec_shapes.append((prev, int(width)))
        prev = int(width)
    dec_shapes.append((prev, int(model.input_dim)))

    enc_layers: List[nn.Module] = []
    for idx, (new_in, new_out) in enumerate(enc_shapes):
        old = old_enc[idx] if idx < len(old_enc) else None
        linear = nn.Linear(new_in, new_out).to(device) if old is None else _resize_linear(old, new_out, new_in)
        enc_layers.extend([linear, nn.ReLU(inplace=True)])
    new_model.encoder = nn.Sequential(*enc_layers)

    dec_layers: List[nn.Module] = []
    for idx, (new_in, new_out) in enumerate(dec_shapes):
        old = old_dec[idx] if idx < len(old_dec) else None
        linear = nn.Linear(new_in, new_out).to(device) if old is None else _resize_linear(old, new_out, new_in)
        dec_layers.append(linear)
        if idx != len(dec_shapes) - 1:
            dec_layers.append(nn.ReLU(inplace=True))
    new_model.decoder = nn.Sequential(*dec_layers)

    filtered_state = {
        key: _resize_tensor(value.shape, old_state[key]) if key in old_state and old_state[key].shape != value.shape else old_state.get(key, value)
        for key, value in new_model.state_dict().items()
    }
    new_model.load_state_dict(filtered_state, strict=False)
    new_model.hidden_widths = list(widths)
    new_model.width = max(widths)
    new_model.depth = len(widths)
    return new_model


def rebuild_sup_dae_model(model_cls: Type[Any], model: Any, hidden_widths: Sequence[int], device: torch.device) -> Any:
    widths = [int(w) for w in hidden_widths]
    if not widths:
        raise ValueError("hidden_widths must be non-empty")
    new_model = model_cls(
        num_classes=model.num_classes,
        in_channels=model.in_channels,
        img_size=model.img_size,
        width=max(widths),
        depth=len(widths),
    ).to(device)
    new_model.dae = rebuild_unsup_dae_model(type(model.dae), model.dae, widths, device)
    new_model.classifier = _resize_linear(model.classifier, model.num_classes, widths[-1])
    new_model.hidden_widths = list(widths)
    new_model.width = max(widths)
    new_model.depth = len(widths)
    return new_model


def expand_unsup_width(model_cls: Type[Any], model: Any, ex_k: int, max_width: int, device: torch.device) -> Optional[Any]:
    current = infer_hidden_widths(model)
    new_widths = next_staged_widths(current, max_width, ex_k)
    if new_widths == current:
        return None
    return rebuild_unsup_dae_model(model_cls, model, new_widths, device)


def expand_sup_width(model_cls: Type[Any], model: Any, ex_k: int, max_width: int, device: torch.device) -> Optional[Any]:
    current = infer_hidden_widths(model)
    new_widths = next_staged_widths(current, max_width, ex_k)
    if new_widths == current:
        return None
    return rebuild_sup_dae_model(model_cls, model, new_widths, device)


def expand_unsup_depth(model_cls: Type[Any], model: Any, max_depth: int, device: torch.device, min_new_layer_width: int = 10) -> Optional[Any]:
    current = infer_hidden_widths(model)
    if len(current) >= max_depth or not current:
        return None
    if len(set(current)) != 1:
        return None
    if int(current[-1]) <= int(min_new_layer_width):
        return None
    return rebuild_unsup_dae_model(model_cls, model, current + [int(min_new_layer_width)], device)


def expand_sup_depth(model_cls: Type[Any], model: Any, max_depth: int, device: torch.device, min_new_layer_width: int = 10) -> Optional[Any]:
    current = infer_hidden_widths(model)
    if len(current) >= max_depth or not current:
        return None
    if len(set(current)) != 1:
        return None
    if int(current[-1]) <= int(min_new_layer_width):
        return None
    return rebuild_sup_dae_model(model_cls, model, current + [int(min_new_layer_width)], device)


def unsup_total_neurons(model: Any) -> int:
    widths = infer_hidden_widths(model)
    return int(2 * sum(widths))


def sup_total_neurons(model: Any) -> int:
    widths = infer_hidden_widths(model)
    if not widths:
        return 0
    return int(2 * sum(widths) + widths[-1] * int(model.num_classes))


def snapshot_unsup_model(model: Any, state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    st = state if state is not None else model.state_dict()
    return {
        "in_channels": model.in_channels,
        "img_size": model.img_size,
        "hidden_widths": infer_hidden_widths(model),
        "state": copy.deepcopy(st),
    }


def restore_unsup_model(model_cls: Type[Any], snap: Dict[str, Any], device: torch.device) -> Any:
    widths = [int(w) for w in snap["hidden_widths"]]
    mdl = model_cls(
        in_channels=snap.get("in_channels", 3),
        img_size=snap.get("img_size", 32),
        width=max(widths),
        depth=len(widths),
    ).to(device)
    mdl = rebuild_unsup_dae_model(model_cls, mdl, widths, device)
    mdl.load_state_dict(snap["state"], strict=False)
    return mdl


def snapshot_sup_model(model: Any, state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    st = state if state is not None else model.state_dict()
    return {
        "num_classes": model.num_classes,
        "in_channels": model.in_channels,
        "img_size": model.img_size,
        "hidden_widths": infer_hidden_widths(model),
        "state": copy.deepcopy(st),
    }


def restore_sup_model(model_cls: Type[Any], snap: Dict[str, Any], device: torch.device) -> Any:
    widths = [int(w) for w in snap["hidden_widths"]]
    mdl = model_cls(
        num_classes=snap["num_classes"],
        in_channels=snap.get("in_channels", 3),
        img_size=snap.get("img_size", 32),
        width=max(widths),
        depth=len(widths),
    ).to(device)
    mdl = rebuild_sup_dae_model(model_cls, mdl, widths, device)
    mdl.load_state_dict(snap["state"], strict=False)
    return mdl
