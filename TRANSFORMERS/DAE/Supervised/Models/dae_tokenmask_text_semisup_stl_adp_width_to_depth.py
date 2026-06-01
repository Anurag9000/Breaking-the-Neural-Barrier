from __future__ import annotations

import copy
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[4]))

from utils.transformer_mlp_adp import bind_legacy_wrapper


BASE_PATH = Path(__file__).with_name("dae_tokenmask_text_semisup_stl.py").resolve()
sys.path.insert(0, str(BASE_PATH.parent))
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
assert _spec is not None and _spec.loader is not None
baseline_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = baseline_module
_spec.loader.exec_module(baseline_module)

ModelClass = baseline_module.SupTokenMaskSemiDAE


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    width_expansion_patience: int = 10
    depth_expansion_patience: int = 2
    width_stage_margin_patience: int = 10
    width_stage_min_improve_pct: float = 1.0
    ex_k: int = 16
    max_width: int = 4096
    max_depth: int = 16
    max_neurons: int = 5_000_000
    min_new_layer_width: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    max_epochs: int = 300


def _move_batch(batch, device):
    if isinstance(batch, (list, tuple)):
        moved = [item.to(device) if hasattr(item, "to") else item for item in batch]
        return moved
    return batch.to(device)


def _compute_loss(model, batch) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        x = batch[0]
        y = batch[1] if len(batch) > 1 else None
    else:
        x, y = batch, None

    outputs = model(x)
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        raise ValueError("Expected semisupervised token-mask model to return (token_logits, cls_logits)")
    token_logits, cls_logits = outputs

    losses = []
    if isinstance(token_logits, torch.Tensor) and token_logits.dim() == 3:
        losses.append(F.cross_entropy(token_logits.reshape(-1, token_logits.size(-1)), x.reshape(-1)))
    if y is not None and isinstance(cls_logits, torch.Tensor) and cls_logits.dim() == 2:
        losses.append(F.cross_entropy(cls_logits, y))
    if not losses:
        raise ValueError("Could not derive a training loss for dae_tokenmask_text_semisup_stl")
    return sum(losses)


def train_with_early_stopping(model: ModelClass, dl_train, dl_val, acfg: ADPConfig, device, history):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for _ in range(acfg.max_epochs):
        model.train()
        for batch in dl_train:
            batch = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = _compute_loss(model, batch)
            loss.backward()
            if acfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()

        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in dl_val:
                batch = _move_batch(batch, device)
                total += float(_compute_loss(model, batch).item())
                count += 1
        val = total / max(count, 1)
        history.append(val)

        if val < best_val - acfg.delta:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1
        if es_counter >= acfg.patience:
            break

    return best_val, best_state


bind_legacy_wrapper(globals())
