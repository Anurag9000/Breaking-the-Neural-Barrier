import torch
import torch.nn as nn

class LNGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.Wz = nn.Linear(input_dim, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wr = nn.Linear(input_dim, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wn = nn.Linear(input_dim, hidden_size)
        self.Un = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Lz = nn.LayerNorm(hidden_size)
        self.Lr = nn.LayerNorm(hidden_size)
        self.Ln = nn.LayerNorm(hidden_size)
        self.reset_parameters()
    def reset_parameters(self):
        for m in [self.Wz, self.Uz, self.Wr, self.Ur, self.Wn, self.Un]:
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x, h):
        z = torch.sigmoid(self.Lz(self.Wz(x) + self.Uz(h)))
        r = torch.sigmoid(self.Lr(self.Wr(x) + self.Ur(h)))
        n = torch.tanh(self.Ln(self.Wn(x) + self.Un(r * h)))
        return (1 - z) * h + z * n

class RNN_LNGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(LNGRUCell(d, hidden_size))
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
