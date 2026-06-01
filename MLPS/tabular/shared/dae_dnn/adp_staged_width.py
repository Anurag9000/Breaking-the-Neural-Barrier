from __future__ import annotations

from typing import List, Optional, Sequence

from MLPS.tabular.shared.dae_dnn.adp_search import _rebuild_mlp, total_neurons
from MLPS.tabular.shared.dae_dnn.mlp import MLP


def next_staged_widths(hidden_widths: Sequence[int], ex_k: int, max_width: int) -> Optional[List[int]]:
    widths = [int(w) for w in hidden_widths]
    if not widths:
        return None

    steps = max(int(ex_k), 1)
    changed = False
    for _ in range(steps):
        eligible = [idx for idx, width in enumerate(widths) if int(width) < int(max_width)]
        if not eligible:
            break
        target_width = min(int(widths[idx]) for idx in eligible)
        target_idx = next(idx for idx in eligible if int(widths[idx]) == target_width)
        widths[target_idx] += 1
        changed = True

    if not changed:
        return None
    return widths


def expand_width_staged(model: MLP, ex_k: int, max_width: int) -> Optional[MLP]:
    new_hidden = next_staged_widths(model.hidden_widths, ex_k, max_width)
    if new_hidden is None:
        return None
    return _rebuild_mlp(model, new_hidden)


def can_widen_staged(model: MLP, max_width: int, max_neurons: int) -> bool:
    if not model.hidden_widths:
        return False
    if total_neurons(model) >= int(max_neurons):
        return False
    return any(int(width) < int(max_width) for width in model.hidden_widths)
