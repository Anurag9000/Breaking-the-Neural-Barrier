import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path
import importlib.util
import types


def _load_tokenmask_backbone() -> types.ModuleType:
    base = Path(__file__).resolve().parents[2] / "Self-Supervised" / "Models" / "dae_tokenmask_text_stl.py"
    spec = importlib.util.spec_from_file_location("dae_tokenmask_text_stl_backbone", base)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load token-mask text DAE backbone from {base}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_backbone = _load_tokenmask_backbone()
TokenMaskTransformerDAE = _backbone.TokenMaskTransformerDAE  # type: ignore[attr-defined]
token_dae_total_neurons = _backbone.token_dae_total_neurons  # type: ignore[attr-defined]


class SupTokenMaskSemiDAE(nn.Module):
    """
    Semi-supervised token-masking transformer DAE.

    - Backbone: TokenMaskTransformerDAE for masked-token reconstruction.
    - Head: CLS-style classification from the first token's hidden state.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        max_len: int = 128,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = TokenMaskTransformerDAE(
            vocab_size=vocab_size,
            d_model=d_model,
            depth=depth,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            pad_id=pad_id,
        )
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.depth = depth
        self.max_len = max_len
        self.pad_id = pad_id

        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: (B, L)
        Returns:
          logits_tokens: (B, L, V) for reconstruction
          logits_cls: (B, num_classes) for classification
        """
        token_logits, hidden = self.backbone(input_ids)
        # Simple CLS representation: first token's hidden state.
        cls_repr = hidden[:, 0, :]
        cls_logits = self.cls_head(cls_repr)
        return token_logits, cls_logits


def sup_token_dae_total_neurons(d_model: int, depth: int, num_classes: int, vocab_size: int) -> int:
    return token_dae_total_neurons(d_model, depth) + d_model * num_classes

