import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path
import importlib.util
import sys
import types


def _load_groupsparse_backbone() -> types.ModuleType:
    base = Path(__file__).resolve().parents[3] / "unsupervised" / "dae" / "Models" / "dae_groupsparse_mlp_stl.py"
    base_parent = str(base.parent)
    if base_parent not in sys.path:
        sys.path.insert(0, base_parent)
    spec = importlib.util.spec_from_file_location("dae_groupsparse_mlp_stl_backbone", base)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load group-sparse MLP DAE backbone from {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backbone = _load_groupsparse_backbone()
DAEGroupSparseMLP = _backbone.DAEGroupSparseMLP  # type: ignore[attr-defined]
dae_total_neurons = _backbone.dae_total_neurons  # type: ignore[attr-defined]


class SupDAEGroupSparseMLP(nn.Module):
    """
    Group-sparse MLP DAE encoder + classifier head.
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
        self.dae = DAEGroupSparseMLP(
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
        x_rec, z = self.dae(x)
        logits = self.classifier(z)
        return x_rec, logits, z


def sup_dae_total_neurons(width: int, depth: int, num_classes: int) -> int:
    return dae_total_neurons(width, depth) + width * num_classes
