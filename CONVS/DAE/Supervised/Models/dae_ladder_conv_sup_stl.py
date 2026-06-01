import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path
import importlib.util
import types


def _load_ladder_backbone() -> types.ModuleType:
    base = Path(__file__).resolve().parents[2] / "Self-Supervised" / "Models" / "dae_ladder_vae_conv_stl.py"
    spec = importlib.util.spec_from_file_location("dae_ladder_vae_conv_stl_backbone", base)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load ladder VAE conv DAE backbone from {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backbone = _load_ladder_backbone()
DAELadderVAEConv = _backbone.DAELadderVAEConv  # type: ignore[attr-defined]
ladder_vae_total_neurons = _backbone.ladder_vae_total_neurons  # type: ignore[attr-defined]


class SupDAELadderConv(nn.Module):
    """
    Ladder-style conv DAE encoder + classifier head.

    For now this wraps DAELadderVAEConv and adds a global-average-pooling
    classifier on the mean latent representation.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        img_size: int = 32,
        width: int = 64,
        depth: int = 4,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.dae = DAELadderVAEConv(
            in_channels=in_channels,
            img_size=img_size,
            width=width,
            depth=depth,
            latent_dim=latent_dim,
        )
        self.num_classes = num_classes
        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.img_size = img_size

        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # DAELadderVAEConv follows DAEVAEConv: returns (x_rec, mu, logvar).
        x_rec, mu, logvar = self.dae(x)
        logits = self.classifier(mu)
        return x_rec, logits, mu


def sup_dae_ladder_total_neurons(width: int, depth: int, latent_dim: int, num_classes: int) -> int:
    return ladder_vae_total_neurons(width, depth, latent_dim) + latent_dim * num_classes

