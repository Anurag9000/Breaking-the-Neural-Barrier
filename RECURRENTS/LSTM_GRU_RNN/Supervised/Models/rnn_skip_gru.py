import torch
import torch.nn as nn

class SkipGRUCell(nn.Module):
    """GRU with learnable skip gate s_t to blend prev state.
    s = sigmoid(Ws[x,h])
    h~ = GRU(x,h)
    h' = s * h~ + (1 - s) * h
    """
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_size)
        self.s = nn.Linear(input_dim + hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.s.weight); nn.init.zeros_(self.s.bias)
    def forward(self, x, h):
        h_tilde = self.gru(x, h)
        s = torch.sigmoid(self.s(torch.cat([x, h], dim=-1)))
        return s * h_tilde + (1 - s) * h

class RNN_SkipGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(SkipGRUCell(d, hidden_size))
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
