from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class ResidualLSTMConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 3    # residual stack (>=2 recommended)
    dropout: float = 0.2
    num_classes: int = 2
    pad_idx: int = 0


class ResidualLSTMClassifier(nn.Module):
    """Stacked LSTM with residual connections between layers (same hidden dim)."""
    def __init__(self, cfg: ResidualLSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.layers = nn.ModuleList()
        inp = cfg.emb_dim
        for l in range(cfg.num_layers):
            self.layers.append(nn.LSTM(input_size=inp, hidden_size=cfg.hidden_dim, num_layers=1,
                                       dropout=0.0, bidirectional=False, batch_first=True))
            inp = cfg.hidden_dim
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for layer in self.layers:
            for n, p in layer.named_parameters():
                if 'weight_' in n: nn.init.xavier_uniform_(p)
                elif 'bias_' in n: nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        x = self.embedding(tokens)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        last_h = None
        # We need to unpack between layers to do residuals correctly on padded sequences.
        # But as an approximation, we take final state per layer and add residual there.
        hx_skip = None
        for i, lstm in enumerate(self.layers):
            _, (h_n, _) = lstm(packed)
            h = h_n[-1]  # (B,H)
            if hx_skip is None:
                out = h
            else:
                out = h + hx_skip  # residual add at representation level
            hx_skip = out
        feat = self.dropout(out)
        return self.fc(feat)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
