from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class PeepholeLSTMConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 2
    pad_idx: int = 0


class PeepholeLSTMCell(nn.Module):
    """LSTMCell with peephole connections (C-to-gates)."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, 4*hidden_size)
        self.U = nn.Linear(hidden_size, 4*hidden_size, bias=False)
        # peephole weights are diagonal per gate, implemented as elementwise parameters
        self.p_i = nn.Parameter(torch.zeros(hidden_size))
        self.p_f = nn.Parameter(torch.zeros(hidden_size))
        self.p_o = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.W, self.U]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.p_i); nn.init.zeros_(self.p_f); nn.init.zeros_(self.p_o)

    def forward(self, x, state):
        h, c = state
        gates = self.W(x) + self.U(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i + self.p_i * c)
        f = torch.sigmoid(f + self.p_f * c)
        g = torch.tanh(g)
        c_next = f * c + i * g
        o = torch.sigmoid(o + self.p_o * c_next)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class PeepholeLSTMClassifier(nn.Module):
    """Many-to-one classifier using custom peephole LSTM stacked L times.
    Processes padded sequences without packing; masks time steps by lengths.
    """
    def __init__(self, cfg: PeepholeLSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.layers = nn.ModuleList([PeepholeLSTMCell(cfg.emb_dim if l==0 else cfg.hidden_dim, cfg.hidden_dim)
                                     for l in range(cfg.num_layers)])
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        emb = self.embedding(tokens)  # (B,T,E)
        B, T, _ = emb.shape
        device = emb.device
        h = [emb.new_zeros(B, self.cfg.hidden_dim) for _ in self.layers]
        c = [emb.new_zeros(B, self.cfg.hidden_dim) for _ in self.layers]
        last = None
        for t in range(T):
            x = emb[:, t]
            for l, cell in enumerate(self.layers):
                h_l, c_l = cell(x, (h[l], c[l]))
                # mask out positions beyond sequence length
                mask = (t < lengths).float().unsqueeze(-1)
                h[l] = h_l * mask + h[l] * (1 - mask)
                c[l] = c_l * mask + c[l] * (1 - mask)
                x = h[l]
            last = h[-1]
        feat = self.dropout(last)
        return self.fc(feat)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
