import torch
import torch.nn as nn

class SRULayer(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.w = nn.Linear(input_dim, 3*hidden_size, bias=True)
        self.proj = nn.Identity() if input_dim == hidden_size else nn.Linear(input_dim, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.w.weight); nn.init.zeros_(self.w.bias)
        if isinstance(self.proj, nn.Linear): nn.init.xavier_uniform_(self.proj.weight)
    def forward(self, x):
        # x: (B,T,D)
        B,T,D = x.size()
        z, f, r = self.w(x).chunk(3, dim=-1)
        z = torch.tanh(z)
        f = torch.sigmoid(f)
        r = torch.sigmoid(r)
        c_prev = torch.zeros(B, self.proj.out_features if isinstance(self.proj, nn.Linear) else D, device=x.device)
        outs = []
        for t in range(T):
            c = f[:, t, :] * c_prev + (1 - f[:, t, :]) * z[:, t, :]
            h = r[:, t, :] * c + (1 - r[:, t, :]) * self.proj(x[:, t, :])
            outs.append(h)
            c_prev = c
        return torch.stack(outs, 1)

class RNN_SRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(SRULayer(d, hidden_size))
            d = hidden_size
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.drop(x)
        return self.head(x[:, -1, :])
