import torch
import torch.nn as nn

class UGRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.z = nn.Linear(input_dim + hidden_size, hidden_size)
        self.c = nn.Linear(input_dim + hidden_size, hidden_size)
        self.reset_parameters()
    def reset_parameters(self):
        for m in (self.z, self.c):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x, h):
        concat = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.z(concat))
        cand = torch.tanh(self.c(concat))
        h_new = (1 - z) * h + z * cand
        return h_new

class RNN_UGRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        d = input_dim
        for _ in range(num_layers):
            self.layers.append(UGRNNCell(d, hidden_size))
            d = hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B, T, D = x.size()
        out = x
        for layer in self.layers:
            h = torch.zeros(B, layer.z.out_features, device=x.device)
            seq = []
            for t in range(T):
                h = layer(out[:, t, :], h)
                seq.append(h)
            out = torch.stack(seq, dim=1)
            out = self.drop(out)
        return self.head(out[:, -1, :])
