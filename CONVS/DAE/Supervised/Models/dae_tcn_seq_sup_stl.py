import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path
import importlib.util
import types


def _load_tcn_backbone() -> types.ModuleType:
    """
    Dynamically load the unsupervised TCN sequence DAE backbone.
    """
    base = Path(__file__).resolve().parents[2] / "Self-Supervised" / "Models" / "dae_tcn_seq_stl.py"
    spec = importlib.util.spec_from_file_location("dae_tcn_seq_stl_backbone", base)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"Could not load TCN DAE backbone from {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backbone = _load_tcn_backbone()
DAETCNSeq = _backbone.DAETCNSeq  # type: ignore[attr-defined]
tcn_total_neurons = _backbone.tcn_total_neurons  # type: ignore[attr-defined]


class SupDAETCNSeq(nn.Module):
    """
    Supervised temporal DAE encoder + sequence classifier.

    Backbone:
        - DAETCNSeq taking (B, C, L) sequences and returning (recon, latent).

    Head:
        - Global average pooling over time + linear classifier to num_classes.
    """

    def __init__(self, num_classes: int, in_channels: int = 1, width: int = 64, depth: int = 4):
        super().__init__()
        self.dae = DAETCNSeq(in_channels=in_channels, width=width, depth=depth)
        self.num_classes = num_classes
        self.width = width
        self.depth = depth

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(width, num_classes)

    @property
    def in_channels(self) -> int:
        return self.dae.in_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_rec, z = self.dae(x)
        # z: (B, C, L) with C==width; pool over time L.
        feat = self.gap(z).view(z.size(0), -1)
        logits = self.classifier(feat)
        return x_rec, logits


def sup_dae_total_neurons(width: int, depth: int, num_classes: int) -> int:
    return tcn_total_neurons(width, depth) + width * num_classes

