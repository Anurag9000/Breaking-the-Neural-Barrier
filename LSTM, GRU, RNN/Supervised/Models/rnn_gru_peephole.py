import torch
import torch.nn as nn

class PeepholeGRUCell(nn.Module):
    """GRU with elementwise peephole connections to gates (additional diagonal terms)."""
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.Wz = nn.Linear(input_dim, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wr = nn.Linear(input_dim, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wn = nn.Linear(input_dim, hidden_size)
        self.Un = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pz = nn.Parameter(torch.zeros(hidden_size))
        self.pr = nn.Parameter(torch.zeros(hidden_size))
        self.pn = nn.Parameter(torch.zeros(hidden_size))
        for m in [self.Wz, self.Uz, self.Wr, self.Ur, self.Wn, self.Un]:
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x, h):
        z = torch.sigmoid(self.Wz(x) + self.Uz(h) + self.pz * h)
        r = torch.sigmoid(self.Wr(x) + self.Ur(h) + self.pr * h)
        n = torch.tanh(self.Wn(x) + self.Un(r * h) + self.pn * (r * h))
        return (1 - z) * h + z * n

class RNN_GRU_Peephole(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(PeepholeGRUCell(d, hidden_size))
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
