from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class LSTMTaggerConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    num_tags: int = 10
    pad_idx: int = 0
    bidirectional: bool = False


class LSTMTagger(nn.Module):
    """Vanilla LSTM (many-to-many aligned): per-timestep tagging.
    Returns unnormalized tag logits for each timestep.
    """
    def __init__(self, cfg: LSTMTaggerConfig):
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
        self.classifier = nn.Linear(out_dim, cfg.num_tags)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for name, p in self.lstm.named_parameters():
            if "weight_" in name:
                nn.init.xavier_uniform_(p)
            elif "bias_" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        emb = self.embedding(tokens)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        logits = self.classifier(out)  # (B, T, num_tags)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
