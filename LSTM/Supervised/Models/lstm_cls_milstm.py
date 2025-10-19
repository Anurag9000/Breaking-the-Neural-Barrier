from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class MILSTMConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 2
    pad_idx: int = 0


class MILSTMCell(nn.Module):
    """Multiplicative Integration LSTM cell.
    Replaces affine add with elementwise multiplicative mixing.
    i = σ((W_i x) ⊙ (U_i h) + b_i), etc.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(input_size, 4 * hidden_size)
        self.U = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight); nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.zeros_(self.b)

    def forward(self, x, state):
        h, c = state
        Wx = self.W(x)
        Uh = self.U(h)
        gates = (Wx * Uh) + self.b
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        c_next = f * c + i * g
        o = torch.sigmoid(o)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class MILSTMClassifier(nn.Module):
    def __init__(self, cfg: MILSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.layers = nn.ModuleList([
            MILSTMCell(cfg.emb_dim if l == 0 else cfg.hidden_dim, cfg.hidden_dim)
            for l in range(cfg.num_layers)
        ])
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        emb = self.embedding(tokens)
        B, T, _ = emb.shape
        h = [emb.new_zeros(B, self.cfg.hidden_dim) for _ in self.layers]
        c = [emb.new_zeros(B, self.cfg.hidden_dim) for _ in self.layers]
        for t in range(T):
            x = emb[:, t]
            for l, cell in enumerate(self.layers):
                h_l, c_l = cell(x, (h[l], c[l]))
                mask = (t < lengths).float().unsqueeze(-1)
                h[l] = h_l * mask + h[l] * (1 - mask)
                c[l] = c_l * mask + c[l] * (1 - mask)
                x = h[l]
        feat = self.dropout(h[-1])
        return self.fc(feat)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
