from __future__ import annotations

from typing import Any, Iterable, Tuple


def _get_attr(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None or not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _first_present(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = _get_attr(obj, name, None)
        if value is not None:
            return value
    return default


def infer_adp_width(model: Any, default: int = 0) -> int:
    candidates = (
        "width",
        "hidden_dim",
        "hidden",
        "dim",
        "d_model",
        "embed_dim",
        "hidden_size",
        "channels",
        "feature_dim",
        "latent_dim",
        "z_dim",
    )
    value = _first_present(model, candidates, None)
    if value is None:
        value = _first_present(getattr(model, "cfg", None), candidates, None)
    if value is None and hasattr(model, "hidden_widths"):
        widths = getattr(model, "hidden_widths")
        if widths:
            return int(max(int(w) for w in widths))
    return int(value) if value is not None else int(default)


def infer_adp_depth(model: Any, default: int = 1) -> int:
    candidates = (
        "depth",
        "num_layers",
        "layers",
        "n_layers",
        "num_blocks",
        "block_depth",
        "encoder_depth",
        "decoder_depth",
    )
    value = _first_present(model, candidates, None)
    if value is None:
        value = _first_present(getattr(model, "cfg", None), candidates, None)
    if value is None and hasattr(model, "hidden_widths"):
        widths = getattr(model, "hidden_widths")
        if widths is not None:
            return int(len(widths))
    return int(value) if value is not None else int(default)


def infer_adp_shape(model: Any, default_width: int = 0, default_depth: int = 1) -> Tuple[int, int]:
    return infer_adp_width(model, default_width), infer_adp_depth(model, default_depth)


def can_expand_width(model: Any, cfg: Any, ex_k_attr: str = "ex_k", max_width_attr: str = "max_width") -> bool:
    width = infer_adp_width(model)
    ex_k = int(getattr(cfg, ex_k_attr, 1))
    max_width = int(getattr(cfg, max_width_attr, width))
    return width + ex_k <= max_width


def can_expand_depth(model: Any, cfg: Any, max_depth_attr: str = "max_depth") -> bool:
    depth = infer_adp_depth(model)
    max_depth = int(getattr(cfg, max_depth_attr, depth))
    return depth + 1 <= max_depth
