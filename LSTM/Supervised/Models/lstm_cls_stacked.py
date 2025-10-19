from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class StackedLSTMConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2  # stacked (>=2)
    dropout: float = 0.2
    num_classes: int = 2
    pad_idx: int = 0


class StackedLSTMClassifier(nn.Module):
    """Stacked (multi-layer) LSTM many-to-one classifier.
    Enforces num_layers >= 2; uses final hidden state of top layer.
    """
    def __init__(self, cfg: StackedLSTMConfig):
        super().__init__()
        assert cfg.num_layers >= 2, "StackedLSTMClassifier requires num_layers >= 2"
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for name, p in self.lstm.named_parameters():
            if "weight_" in name:
                nn.init.xavier_uniform_(p)
            elif "bias_" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        emb = self.embedding(tokens)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        feat = h_n[-1]
        feat = self.dropout(feat)
        return self.fc(feat)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
