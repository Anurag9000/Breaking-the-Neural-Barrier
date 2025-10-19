import torch
import torch.nn as nn

class DilatedRNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, dilation: int, nonlinearity='tanh'):
        super().__init__()
        self.cell = nn.RNNCell(input_dim, hidden_size, nonlinearity=nonlinearity)
        self.dilation = max(1, int(dilation))
    def forward(self, x):
        B,T,D = x.size()
        h = torch.zeros(B, self.cell.hidden_size, device=x.device)
        outs = []
        for t in range(T):
            if (t % self.dilation) == 0:
                h = self.cell(x[:, t, :], h)
            outs.append(h)
        return torch.stack(outs, 1)

class RNN_Dilated(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, base:int=2, dropout: float=0.1, nonlinearity='tanh'):
        super().__init__()
        layers = []
        d = input_dim
        for i in range(num_layers):
            layers.append(DilatedRNNLayer(d, hidden_size, dilation=(base ** i), nonlinearity=nonlinearity))
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
