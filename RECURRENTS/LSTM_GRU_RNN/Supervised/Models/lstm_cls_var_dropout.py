from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class VarDropLSTMConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.5   # variational (locked) dropout prob
    num_classes: int = 2
    pad_idx: int = 0


class LockedDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        # x: (B, T, D)
        mask = x.new_ones(x.size(0), 1, x.size(2))
        mask = torch.nn.functional.dropout(mask, p=self.p, training=True)
        return x * mask


class VarDropLSTMClassifier(nn.Module):
    """Variational (locked) dropout LSTM: same dropout mask across time steps.
    Applies locked dropout on embeddings and between LSTM layers.
    """
    def __init__(self, cfg: VarDropLSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.lockdrop = LockedDropout(cfg.dropout)
        self.lstm = nn.LSTM(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=0.0,  # we handle dropout ourselves
            bidirectional=False,
            batch_first=True,
        )
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for n,p in self.lstm.named_parameters():
            if 'weight_' in n: nn.init.xavier_uniform_(p)
            elif 'bias_' in n: nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        x = self.embedding(tokens)
        x = self.lockdrop(x)
        # pack -> run first layer
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, (h, c) = self.lstm(x_packed)
        # final state from top layer
        feat = h[-1]
        feat = torch.nn.functional.dropout(feat, p=self.cfg.dropout, training=self.training)
        return self.fc(feat)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
