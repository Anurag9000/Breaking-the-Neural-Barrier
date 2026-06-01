import torch
import torch.nn as nn

class IndGRUCell(nn.Module):
    """Independently recurrent GRU: per-unit recurrent params (diagonal Un, Ur, Uz)."""
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.Wz = nn.Linear(input_dim, hidden_size)
        self.Wr = nn.Linear(input_dim, hidden_size)
        self.Wn = nn.Linear(input_dim, hidden_size)
        self.uz = nn.Parameter(torch.Tensor(hidden_size))
        self.ur = nn.Parameter(torch.Tensor(hidden_size))
        self.un = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()
    def reset_parameters(self):
        for m in (self.Wz, self.Wr, self.Wn):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for v in (self.uz, self.ur, self.un):
            nn.init.uniform_(v, a=0.0, b=1.0)
    def forward(self, x, h):
        z = torch.sigmoid(self.Wz(x) + h * self.uz)
        r = torch.sigmoid(self.Wr(x) + h * self.ur)
        n = torch.tanh(self.Wn(x) + (r * h) * self.un)
        return (1 - z) * h + z * n

class RNN_IndGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(IndGRUCell(d, hidden_size))
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
