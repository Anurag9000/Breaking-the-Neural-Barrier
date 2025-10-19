import torch
import torch.nn as nn

class LNRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_size, nonlinearity='tanh'):
        super().__init__()
        self.lin = nn.Linear(input_dim + hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.nonlinearity = nonlinearity
        nn.init.xavier_uniform_(self.lin.weight); nn.init.zeros_(self.lin.bias)
    def forward(self, x, h):
        pre = self.lin(torch.cat([x, h], dim=-1))
        pre = self.ln(pre)
        if self.nonlinearity == 'relu':
            return torch.relu(pre)
        else:
            return torch.tanh(pre)

class RNN_Vanilla_LN(nn.Module):
    def __init__(self, input_dim:int, hidden_size:int, num_layers:int, num_classes:int, dropout:float=0.1, nonlinearity='tanh'):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(LNRNNCell(d, hidden_size, nonlinearity))
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
