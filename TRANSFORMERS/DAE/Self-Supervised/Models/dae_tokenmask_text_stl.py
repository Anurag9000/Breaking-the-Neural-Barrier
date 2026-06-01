import math
from typing import Tuple

import torch
import torch.nn as nn


class TokenMaskTransformerDAE(nn.Module):
    """
    Simple transformer encoder for masked token reconstruction (BERT-style DAE).

    - Input: sequences of token ids (B, L).
    - Internals: token embedding + positional embedding + TransformerEncoder.
    - Output: per-token logits over vocabulary (B, L, V).

    Masking (replacing some tokens with a special [MASK] id) is handled in the
    training loop; this module just produces logits.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        max_len: int = 128,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.depth = depth
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len
        self.pad_id = pad_id

        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: (B, L)
        returns:
          logits: (B, L, V)
          hidden: (B, L, D)
        """
        B, L = input_ids.shape
        device = input_ids.device
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)

        # Padding mask: True means "ignore"
        pad_mask = input_ids.eq(self.pad_id)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        logits = self.head(x)
        return logits, x


def token_dae_total_neurons(d_model: int, depth: int) -> int:
    """
    Simple proxy: d_model * depth.
    """
    return int(d_model * depth)

