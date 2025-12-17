import torch
import torch.nn as nn
from pathlib import Path
import importlib.util
import types
from typing import Tuple


def _load_mae_backbone() -> types.ModuleType:
    base = Path(__file__).resolve().parents[2] / "Self-Supervised" / "Models" / "dae_mae_vit_stl.py"
    spec = importlib.util.spec_from_file_location("dae_mae_vit_stl_backbone_sup", base)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MAEViT backbone from {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


_backbone = _load_mae_backbone()
MAEViT = _backbone.MAEViT  # type: ignore[attr-defined]
mae_total_neurons = _backbone.mae_total_neurons  # type: ignore[attr-defined]


class SupMAEViT(nn.Module):
    """
    MAE-style ViT DAE encoder + classifier head.
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 3,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.dae = MAEViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mask_ratio=mask_ratio,
        )
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # MAEViT forward returns reconstructed image and [CLS] embedding.
        x_rec, cls_tok = self.dae(x)
        logits = self.cls_head(cls_tok)
        return x_rec, logits


def sup_mae_total_neurons(embed_dim: int, depth: int, num_classes: int) -> int:
    return mae_total_neurons(embed_dim, depth) + embed_dim * num_classes

