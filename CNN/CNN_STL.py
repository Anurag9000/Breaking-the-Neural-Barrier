"""
cnn_stl.py
==========

Standard convolutional neural network (STL - Single Task Learning) for CIFAR-100.

Purpose:
    Baseline model that trains a separate CNN for each task without knowledge transfer.

Structure:
    - Configurable number of conv blocks (depth)
    - Configurable channels per block (width)
    - Each block: Conv(3×3, stride=1, padding=1) → BatchNorm → ReLU
    - Optional MaxPool(2×2) after selected blocks (by index)
    - Global Average Pool → Linear head to 100 classes

Notes:
    • Mask/bounds/constraint logic is intentionally omitted (mask-free).
    • Designed as a CNN counterpart to your MLP `FullyConnectedNet` baseline.

Logging:
    • On init, logs depth, widths list, last width, pooling indices, neuron count.
    • Helpers provided to compute and log per-epoch stats with consistent keys.
"""

from __future__ import annotations
import logging
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────
# Logger setup (safe default console handler; avoids duplicates)
# ─────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_ch)

# ─────────────────────────────────────────────────────────────────────────
# Small helpers (reused by trainers for structured logs)
# ─────────────────────────────────────────────────────────────────────────
def stl_total_neurons(model: "ConvNetSTL") -> int:
    """
    Approximate 'neuron' count for parity with ADP plots:
    sum(conv out_channels over all blocks) + head fan-in (= last width).
    """
    return int(sum(model.widths_list) + model.last_width)

def stl_arch_summary(model: "ConvNetSTL") -> Dict[str, Any]:
    """Return a dict you can directly splat into logs/JSON."""
    return dict(
        neurons=stl_total_neurons(model),
        depth=model.depth,
        last_width=model.last_width,
        widths=model.widths_list,
        pooling_indices=sorted(model.pooling_indices),
        num_classes=model.num_classes,
    )

def stl_log_epoch(
    model: "ConvNetSTL",
    *,
    phase: str,
    epoch: int,
    max_epochs: int,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    best_val: Optional[float] = None,
    best_acc: Optional[float] = None,
    bad: Optional[int] = None,
    patience: Optional[int] = None,
    global_epoch: Optional[int] = None,
) -> None:
    """
    Convenience function to emit a single, consistent per-epoch log line.
    Use it in your trainer:
        stl_log_epoch(model, phase="Inner", epoch=e, max_epochs=E, ...)

    All numeric args are optional; omitted ones are skipped in the message.
    """
    pieces = [
        f"[{phase}][epoch {epoch}/{max_epochs}" + (f" | global_epoch={global_epoch}" if global_epoch is not None else "") + "]"
    ]
    if train_loss is not None: pieces.append(f"train_loss={train_loss:.4f}")
    if val_loss   is not None: pieces.append(f"val_loss={val_loss:.4f}")
    if val_acc    is not None: pieces.append(f"val_acc={val_acc:.4f}")
    if best_val   is not None: pieces.append(f"best_val={best_val:.4f}")
    if best_acc   is not None: pieces.append(f"best_acc={best_acc:.4f}")
    if bad is not None and patience is not None: pieces.append(f"bad={bad}/{patience}")

    # arch stats (always include)
    pieces.append(
        f"neurons={stl_total_neurons(model)} | depth={model.depth} | last_width={model.last_width} | widths={model.widths_list}"
    )
    logger.info(" | ".join(pieces))

# ─────────────────────────────────────────────────────────────────────────
# Model blocks
# ─────────────────────────────────────────────────────────────────────────
class ConvBNReLU(nn.Module):
    """Conv(3×3, s=1, p=1) → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

# ─────────────────────────────────────────────────────────────────────────
# Baseline CNN
# ─────────────────────────────────────────────────────────────────────────
class ConvNetSTL(nn.Module):
    """
    Convolutional baseline with configurable depth (blocks) and width (channels).
    Mirrors the intent of `FullyConnectedNet` for images.
    """
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 100,
        *,
        width: Optional[int] = None,
        depth: int = 4,
        pooling_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            input_channels: Input channels (3 for RGB).
            num_classes: Number of target classes (100 for CIFAR-100).
            width: Channels in each block; defaults to 64 if None.
            depth: Number of ConvBNReLU blocks (≥1).
            pooling_indices: Block indices after which to apply MaxPool(2×2).
        """
        super().__init__()

        if width is None:
            width = 64
        if depth < 1:
            raise ValueError("`depth` must be ≥ 1")

        self.num_classes = int(num_classes)
        self.pooling_indices = set(pooling_indices or [1, 3])  # typical for 32×32 inputs

        # build conv stack
        convs: List[nn.Module] = []
        in_ch = input_channels
        for _ in range(depth):
            convs.append(ConvBNReLU(in_ch, width))
            in_ch = width
        self.convs = nn.ModuleList(convs)

        # head
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(width, num_classes)

        # ── log architecture summary on init
        logger.info(
            "[Init ConvNetSTL] "
            f"in_ch={input_channels} | num_classes={num_classes} | "
            f"depth={self.depth} | last_width={self.last_width} | widths={self.widths_list} | "
            f"pooling_indices={sorted(self.pooling_indices)} | neurons={stl_total_neurons(self)}"
        )

    # ---- properties for uniform access/logging ----
    @property
    def depth(self) -> int:
        return len(self.convs)

    @property
    def last_width(self) -> int:
        # same as head.in_features
        return self.convs[-1].bn.num_features

    @property
    def widths_list(self) -> List[int]:
        return [blk.bn.num_features for blk in self.convs if isinstance(blk, ConvBNReLU)]

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.convs):
            x = block(x)
            if i in self.pooling_indices:
                x = F.max_pool2d(x, 2)
        x = self.gap(x).flatten(1)
        return self.head(x)

    # ---- helpers (parity with MLP API) ----
    def get_all_shared_weights(self) -> list[torch.Tensor]:
        """Return weight tensors of all conv layers (shared feature extractor)."""
        weights = [m.conv.weight for m in self.convs if isinstance(m, ConvBNReLU)]
        logger.debug(f"[ConvNetSTL] Collected weights from {len(weights)} conv blocks.")
        return weights

    # Optional: emit a one-line summary at any point (useful at checkpoints)
    def log_arch(self, prefix: str = "[ConvNetSTL]") -> None:
        s = stl_arch_summary(self)
        logger.info(
            f"{prefix} depth={s['depth']} | last_width={s['last_width']} | widths={s['widths']} | "
            f"neurons={s['neurons']} | pooling_indices={s['pooling_indices']} | num_classes={s['num_classes']}"
        )
