from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class BiLSTMConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 2
    pad_idx: int = 0


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier (many-to-one).
    Concatenates forward+backward final states of the top layer.
    """
    def __init__(self, cfg: BiLSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_dim * 2, cfg.num_classes)
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
        # h_n shape: (num_layers*2, B, H)
        h_f = h_n[-2]
        h_b = h_n[-1]
        feat = torch.cat([h_f, h_b], dim=-1)
        feat = self.dropout(feat)
        return self.fc(feat)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
