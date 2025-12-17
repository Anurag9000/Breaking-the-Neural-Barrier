import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path
import importlib.util
import types


def _load_sparse_backbone() -> types.ModuleType:
    """
    Dynamically load the unsupervised sparse MLP DAE backbone.
    """
    base = Path(__file__).resolve().parents[2] / "Self-Supervised" / "Models" / "dae_sparse_mlp_stl.py"
    spec = importlib.util.spec_from_file_location("dae_sparse_mlp_stl_backbone", base)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sparse MLP DAE backbone from {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backbone = _load_sparse_backbone()
DAESparseMLP = _backbone.DAESparseMLP  # type: ignore[attr-defined]
dae_total_neurons = _backbone.dae_total_neurons  # type: ignore[attr-defined]


class SupDAESparseMLP(nn.Module):
    """
    Sparse MLP DAE encoder + classifier head.

    - Backbone: DAESparseMLP operating on flattened CIFAR images.
    - Head: linear classifier on the latent MLP representation.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        img_size: int = 32,
        width: int = 512,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.dae = DAESparseMLP(
            in_channels=in_channels,
            img_size=img_size,
            width=width,
            depth=depth,
        )
        self.num_classes = num_classes
        self.width = width
        self.depth = depth
        self.in_channels = in_channels
        self.img_size = img_size

        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_rec: reconstructed image
            logits: class logits
            z: latent representation (for sparsity penalty)
        """
        x_rec, z = self.dae(x)
        logits = self.classifier(z)
        return x_rec, logits, z


def sup_dae_total_neurons(width: int, depth: int, num_classes: int) -> int:
    return dae_total_neurons(width, depth) + width * num_classes

