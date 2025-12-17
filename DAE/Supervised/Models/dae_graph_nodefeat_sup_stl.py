import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path
import importlib.util
import types


def _load_graph_backbone() -> types.ModuleType:
    """Load unsupervised graph node-feature DAE backbone via path import."""
    base = Path(__file__).resolve().parents[2] / "Self-Supervised" / "Models" / "dae_graph_nodefeat_stl.py"
    spec = importlib.util.spec_from_file_location("dae_graph_nodefeat_stl_backbone", base)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load graph DAE backbone from {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backbone = _load_graph_backbone()
DAEGraphNodeFeat = _backbone.DAEGraphNodeFeat  # type: ignore[attr-defined]
graph_node_dae_total_neurons = _backbone.graph_node_dae_total_neurons  # type: ignore[attr-defined]


class SupDAEGraphNodeFeat(nn.Module):
    """Graph node-feature DAE encoder + node classifier head."""

    def __init__(self, in_dim: int, num_classes: int, width: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.dae = DAEGraphNodeFeat(in_dim=in_dim, width=width, depth=depth)
        self.num_classes = num_classes
        self.width = width
        self.depth = depth
        self.classifier = nn.Linear(width, num_classes)

    @property
    def in_dim(self) -> int:
        return self.dae.in_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_rec, z = self.dae(x, adj)
        logits = self.classifier(z)
        return x_rec, logits


def sup_dae_total_neurons(width: int, depth: int, num_classes: int) -> int:
    return graph_node_dae_total_neurons(width, depth) + width * num_classes

