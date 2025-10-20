import math
from typing import Optional
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.size()
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        return x + self.pe(pos)

class BERTEncoder(nn.Module):
    """
    Minimal BERT-style encoder-only Transformer for supervised tasks (CLS token).
    """
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        self.pos = LearnedPositionalEncoding(max_len + 1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.pad_id = pad_id

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, S)
        x = self.tok(tokens)
        B = x.size(0)
        cls_tok = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)
        x = self.pos(x)
        key_padding_mask = (torch.cat([torch.zeros(B, 1, device=tokens.device, dtype=torch.long), tokens], dim=1) == self.pad_id)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x[:, 0])  # CLS
        logits = self.head(x)
        return logits
