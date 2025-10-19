import torch
import torch.nn as nn

class HighwayGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_size)
        self.t = nn.Linear(input_dim + hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.t.weight); nn.init.zeros_(self.t.bias)
    def forward(self, x, h):
        g = self.gru(x, h)
        t = torch.sigmoid(self.t(torch.cat([x, h], dim=-1)))
        return t * g + (1 - t) * h

class RNN_HighwayGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(HighwayGRUCell(d, hidden_size))
            d = hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        B, T, D = x.size()
        out = x
        for cell in self.layers:
            h = torch.zeros(B, self.head.in_features, device=x.device)
            seq = []
            for t in range(T):
                h = cell(out[:, t, :], h)
                seq.append(h)
            out = torch.stack(seq, 1)
            out = self.drop(out)
        return self.head(out[:, -1, :])
