import importlib.util
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


def _load_gaussian_backbone():
    """
    Load the unsupervised Gaussian Conv DAE backbone used for the metric model.

    This mirrors dae_gaussian_conv_sup_stl._load_gaussian_backbone but is
    local to keep this module self-contained.
    """
    base = (
        Path(__file__)
        .resolve()
        .parents[2]
        / "Self-Supervised"
        / "Models"
        / "dae_gaussian_conv_stl.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dae_gaussian_conv_stl_backbone_metric", base
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Gaussian DAE backbone from {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


_backbone = _load_gaussian_backbone()
DAEGaussianConv = _backbone.DAEGaussianConv  # type: ignore[attr-defined]
dae_total_neurons = _backbone.dae_total_neurons  # type: ignore[attr-defined]


class SupDAEGaussianMetricConv(nn.Module):
    """
    Gaussian Conv DAE encoder + metric-learning projection head.

    The backbone is the same as for SupDAEGaussianConv, but instead of a
    classifier we expose a low-dimensional embedding suitable for contrastive
    / metric losses.
    """

    def __init__(
        self,
        proj_dim: int,
        in_channels: int = 3,
        width: int = 64,
        depth: int = 4,
        pool_after=None,
    ):
        super().__init__()
        if pool_after is None:
            pool_after = [2]
        self.dae = DAEGaussianConv(
            in_channels=in_channels, width=width, depth=depth, pool_after=pool_after
        )
        self.width = width
        self.depth = depth
        self.proj_dim = proj_dim

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(width, proj_dim)

    @property
    def in_channels(self) -> int:
        return self.dae.in_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Backbone returns reconstruction + conv feature map.
        x_rec, h = self.dae(x)
        feat = self.gap(h).view(h.size(0), -1)
        z = self.proj(feat)
        return x_rec, z


def sup_metric_dae_total_neurons(width: int, depth: int, proj_dim: int) -> int:
    """
    Capacity proxy: conv DAE neurons plus projection head parameters.
    """
    return dae_total_neurons(width, depth) + width * proj_dim

