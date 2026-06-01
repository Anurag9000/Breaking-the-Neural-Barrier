import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path
import importlib.util
import types


def _load_masking_backbone() -> types.ModuleType:
    """
    Dynamically load the unsupervised masking Conv DAE backbone.

    The Self-Supervised folder uses a hyphen in its name, so we resolve the
    file path explicitly and import via importlib instead of a dotted import.
    """
    base = Path(__file__).resolve().parents[2] / "Self-Supervised" / "Models" / "dae_masking_conv_stl.py"
    spec = importlib.util.spec_from_file_location("dae_masking_conv_stl_backbone", base)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Could not load masking DAE backbone from {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backbone = _load_masking_backbone()
DAEMaskingConv = _backbone.DAEMaskingConv  # type: ignore[attr-defined]
dae_total_neurons = _backbone.dae_total_neurons  # type: ignore[attr-defined]


class SupDAEMaskingConv(nn.Module):
    """
    Masking Conv DAE encoder + classifier head.

    Backbone:
        - DAEMaskingConv, which takes corrupted inputs and reconstructs the
          clean target while producing a latent conv feature map.

    Head:
        - Global average pooling on the latent map followed by a linear
          classifier to `num_classes`.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        width: int = 64,
        depth: int = 4,
        pool_after=None,
    ) -> None:
        super().__init__()
        if pool_after is None:
            pool_after = [2]

        self.dae = DAEMaskingConv(
            in_channels=in_channels,
            width=width,
            depth=depth,
            pool_after=pool_after,
        )
        self.num_classes = num_classes
        self.width = width
        self.depth = depth

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width, num_classes)

    @property
    def in_channels(self) -> int:
        return self.dae.in_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # DAEMaskingConv returns (reconstruction, latent feature map)
        x_rec, h = self.dae(x)
        feat = self.gap(h).view(h.size(0), -1)
        logits = self.classifier(feat)
        return x_rec, logits


def sup_dae_total_neurons(width: int, depth: int, num_classes: int) -> int:
    """
    Capacity proxy for supervised masking DAE:
    conv backbone neurons + classifier parameters.
    """
    return dae_total_neurons(width, depth) + width * num_classes
