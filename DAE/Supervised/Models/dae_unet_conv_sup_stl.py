import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path
import importlib.util
import types


def _load_unet_backbone() -> types.ModuleType:
    """
    Dynamically load the unsupervised U-Net Conv DAE backbone.

    The Self-Supervised directory name contains a hyphen, so we cannot rely
    on a plain dotted import. Instead we resolve the path and import via
    importlib.
    """
    base = Path(__file__).resolve().parents[2] / "Self-Supervised" / "Models" / "dae_unet_conv_stl.py"
    spec = importlib.util.spec_from_file_location("dae_unet_conv_stl_backbone", base)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Could not load U-Net DAE backbone from {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backbone = _load_unet_backbone()
DAEUNetConv = _backbone.DAEUNetConv  # type: ignore[attr-defined]
dae_total_neurons = _backbone.dae_total_neurons  # type: ignore[attr-defined]


class SupDAEUNetConv(nn.Module):
    """
    Supervised U-Net-style DAE encoder + classifier head.

    Backbone:
        - DAEUNetConv, mapping noisy inputs to reconstructions and providing
          a bottleneck feature map.

    Head:
        - Global average pooling over the bottleneck + linear classifier to
          num_classes.

    Note: The plan mentions a segmentation head; for CIFAR in this repo we
    attach an image-level classifier head, which is the setting used in the
    other supervised DAEs.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        width: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.dae = DAEUNetConv(in_channels=in_channels, width=width, depth=depth)
        self.num_classes = num_classes
        self.width = width
        self.depth = depth

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width, num_classes)

    @property
    def in_channels(self) -> int:
        return self.dae.in_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_rec, bottleneck = self.dae(x)
        feat = self.gap(bottleneck).view(bottleneck.size(0), -1)
        logits = self.classifier(feat)
        return x_rec, logits


def sup_dae_total_neurons(width: int, depth: int, num_classes: int) -> int:
    """
    Capacity proxy: backbone neurons + classifier parameters.
    """
    return dae_total_neurons(width, depth) + width * num_classes

