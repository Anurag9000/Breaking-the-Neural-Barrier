import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class LSTMClassifierConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 2
    pad_idx: int = 0
    bidirectional: bool = False


class LSTMClassifier(nn.Module):
    """Vanilla LSTM (many-to-one): classification from final hidden state.
    Uses packed sequences to be robust to variable lengths.
    """
    def __init__(self, cfg: LSTMClassifierConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )
        out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(out_dim, cfg.num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for name, param in self.lstm.named_parameters():
            if "weight_" in name:
                nn.init.xavier_uniform_(param)
            elif "bias_" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor) -> torch.Tensor:
        # tokens: (B, T), lengths: (B,)
        emb = self.embedding(tokens)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        # Use final hidden state from the top layer (concat directions if bidi)
        if self.cfg.bidirectional:
            # h_n shape: (num_layers*2, B, H). Take the last layer's two directions and concat.
            h_f = h_n[-2]  # forward of last layer
            h_b = h_n[-1]  # backward of last layer
            feat = torch.cat([h_f, h_b], dim=-1)
        else:
            feat = h_n[-1]
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
