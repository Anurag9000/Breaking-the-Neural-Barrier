import torch
import torch.nn as nn

class IndRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_size, nonlinearity='relu'):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden_size)
        self.u = nn.Parameter(torch.Tensor(hidden_size))
        self.nonlinearity = nonlinearity
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.inp.weight)
        nn.init.zeros_(self.inp.bias)
        nn.init.uniform_(self.u, a=0.0, b=1.0)

    def forward(self, x, h):
        # x: (B, D), h: (B, H)
        pre = self.inp(x) + h * self.u
        if self.nonlinearity == 'relu':
            return torch.relu(pre)
        else:
            return torch.tanh(pre)

class RNN_IndRNN(nn.Module):
    """Stacked IndRNN."""
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1, nonlinearity='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(IndRNNCell(d, hidden_size, nonlinearity))
            d = hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B,T,D)
        B, T, D = x.size()
        h = None
        out = x
        for layer in self.layers:
            h = torch.zeros(B, layer.inp.out_features, device=x.device)
            seq_out = []
            for t in range(T):
                h = layer(out[:, t, :], h)
                seq_out.append(h)
            out = torch.stack(seq_out, dim=1)
            out = self.drop(out)
        return self.head(out[:, -1, :])
