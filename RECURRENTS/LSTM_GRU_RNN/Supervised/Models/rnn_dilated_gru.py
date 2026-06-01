import torch
import torch.nn as nn

class DilatedGRULayer(nn.Module):
    def __init__(self, input_dim, hidden_size, dilation: int):
        super().__init__()
        self.cell = nn.GRUCell(input_dim, hidden_size)
        self.dilation = max(1, int(dilation))
    def forward(self, x):
        # x: (B,T,D)
        B, T, D = x.size()
        h = torch.zeros(B, self.cell.hidden_size, device=x.device)
        outs = []
        for t in range(T):
            if (t % self.dilation) == 0:
                h = self.cell(x[:, t, :], h)
            outs.append(h)
        return torch.stack(outs, 1)

class RNN_DilatedGRU(nn.Module):
    """Stacked GRU with temporal dilation per layer (1,2,4,...)"""
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, base: int=2, dropout: float=0.1):
        super().__init__()
        layers = []
        d = input_dim
        for i in range(num_layers):
            layers.append(DilatedGRULayer(d, hidden_size, dilation=(base ** i)))
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
