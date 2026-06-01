import torch
import torch.nn as nn

class SkipRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_size, nonlinearity='tanh'):
        super().__init__()
        self.inp = nn.Linear(input_dim + hidden_size, hidden_size)
        self.gate = nn.Linear(input_dim + hidden_size, hidden_size)
        self.nonlinearity = nonlinearity
        nn.init.xavier_uniform_(self.inp.weight); nn.init.zeros_(self.inp.bias)
        nn.init.xavier_uniform_(self.gate.weight); nn.init.zeros_(self.gate.bias)
    def forward(self, x, h):
        concat = torch.cat([x, h], dim=-1)
        g = torch.sigmoid(self.gate(concat))
        pre = self.inp(concat)
        if self.nonlinearity == 'relu':
            cand = torch.relu(pre)
        else:
            cand = torch.tanh(pre)
        return g * cand + (1 - g) * h

class RNN_SkipRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1, nonlinearity='tanh'):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(SkipRNNCell(d, hidden_size, nonlinearity))
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
