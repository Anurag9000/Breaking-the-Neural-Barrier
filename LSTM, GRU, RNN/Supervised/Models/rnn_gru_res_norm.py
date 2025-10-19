import torch
import torch.nn as nn

class ResNormGRULayer(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True)
        self.proj = nn.Identity() if input_dim == hidden_size else nn.Linear(input_dim, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.ln(out + self.proj(x))
        return out

class RNN_GRU_ResNorm(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float=0.1):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(ResNormGRULayer(d, hidden_size))
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
