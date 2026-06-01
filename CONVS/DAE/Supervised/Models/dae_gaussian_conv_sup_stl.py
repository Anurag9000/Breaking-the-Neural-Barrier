import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path
import importlib.util
import types


def _load_gaussian_backbone() -> types.ModuleType:
    """
    Dynamically load the unsupervised Gaussian DAE backbone.

    The Self-Supervised folder uses a hyphen in its name, which prevents using
    a standard dotted import. We therefore resolve the file path explicitly
    and load the module via importlib.
    """
    base = Path(__file__).resolve().parents[2] / "Self-Supervised" / "Models" / "dae_gaussian_conv_stl.py"
    spec = importlib.util.spec_from_file_location("dae_gaussian_conv_stl_backbone", base)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Could not load Gaussian DAE backbone from {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backbone = _load_gaussian_backbone()
DAEGaussianConv = _backbone.DAEGaussianConv  # type: ignore[attr-defined]
dae_total_neurons = _backbone.dae_total_neurons  # type: ignore[attr-defined]


class SupDAEGaussianConv(nn.Module):
    """
    Gaussian Conv DAE encoder + classifier head.

    - Backbone: DAEGaussianConv, used both for reconstruction and as feature
      extractor (latent from the final conv block).
    - Head: global average pooling over latent feature map followed by a
      linear classifier to num_classes.
    """

    def __init__(self, num_classes: int, in_channels: int = 3, width: int = 64, depth: int = 4, pool_after=None):
        super().__init__()
        if pool_after is None:
            pool_after = [2]
        self.dae = DAEGaussianConv(in_channels=in_channels, width=width, depth=depth, pool_after=pool_after)
        self.num_classes = num_classes
        self.width = width
        self.depth = depth

        # Simple classifier: GAP over last conv feature map + linear layer.
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Infer feature dim from width; backbone uses width channels in last block.
        self.classifier = nn.Linear(width, num_classes)

    @property
    def in_channels(self) -> int:
        return self.dae.in_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # DAEGaussianConv forward returns (x_rec, latent).
        x_rec, h = self.dae(x)
        # h is convolutional feature map; pool and classify.
        feat = self.gap(h).view(h.size(0), -1)
        logits = self.classifier(feat)
        return x_rec, logits


def sup_dae_total_neurons(width: int, depth: int, num_classes: int) -> int:
    # Capacity proxy: DAE conv neurons + classifier params.
    return dae_total_neurons(width, depth) + width * num_classes
