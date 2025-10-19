import torch
import torch.nn as nn

class RHNCell(nn.Module):
    """Recurrent Highway Network cell with L highway transforms per time step."""
    def __init__(self, input_dim, hidden_size, depth=2):
        super().__init__()
        self.depth = depth
        self.H = nn.ModuleList()
        self.T = nn.ModuleList()
        for l in range(depth):
            d_in = input_dim if l == 0 else hidden_size
            self.H.append(nn.Linear(d_in + hidden_size, hidden_size))
            self.T.append(nn.Linear(d_in + hidden_size, hidden_size))
        for m in list(self.H) + list(self.T):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x, h):
        s = h
        for l in range(self.depth):
            d_in = x if l == 0 else 0*s + s  # use s as input thereafter
            concat = torch.cat([d_in, s], dim=-1)
            h_tilde = torch.tanh(self.H[l](concat))
            t = torch.sigmoid(self.T[l](concat))
            s = t * h_tilde + (1 - t) * s
        return s

class RNN_RHN(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, depth: int=2, dropout: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(RHNCell(d, hidden_size, depth))
            d = hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        B,T,D = x.size()
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
