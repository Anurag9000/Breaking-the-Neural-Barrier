from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class LSTMPConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 512   # cell dim
    proj_size: int = 256    # projected hidden dim
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 2
    pad_idx: int = 0


class LSTMPClassifier(nn.Module):
    """LSTM with projection (LSTMP): cell size > projected hidden; many-to-one CLS."""
    def __init__(self, cfg: LSTMPConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=False,
            proj_size=cfg.proj_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.proj_size, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for n, p in self.lstm.named_parameters():
            if 'weight_' in n:
                nn.init.xavier_uniform_(p)
            elif 'bias_' in n:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        emb = self.embedding(tokens)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        feat = h_n[-1]
        return self.fc(self.dropout(feat))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
