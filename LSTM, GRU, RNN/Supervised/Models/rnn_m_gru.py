import torch
import torch.nn as nn

class mGRUCell(nn.Module):
    """Minimal GRU variant: single update gate; candidate uses gated hidden.
    z = sigmoid(Wz[x,h])
    n = tanh(Wn x + (z * (Un h)))
    h' = (1 - z) * h + z * n
    """
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.z = nn.Linear(input_dim + hidden_size, hidden_size)
        self.nx = nn.Linear(input_dim, hidden_size)
        self.nh = nn.Linear(hidden_size, hidden_size, bias=False)
        for m in (self.z, self.nx, self.nh):
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.z.bias); nn.init.zeros_(self.nx.bias)

    def forward(self, x, h):
        z = torch.sigmoid(self.z(torch.cat([x, h], dim=-1)))
        n = torch.tanh(self.nx(x) + self.nh(z * h))
        return (1 - z) * h + z * n

class RNN_mGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(mGRUCell(d, hidden_size))
            d = hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B, T, D = x.size()
        out = x
        for layer in self.layers:
            h = torch.zeros(B, self.head.in_features, device=x.device)
            seq = []
            for t in range(T):
                h = layer(out[:, t, :], h)
                seq.append(h)
            out = torch.stack(seq, dim=1)
            out = self.drop(out)
        return self.head(out[:, -1, :])
