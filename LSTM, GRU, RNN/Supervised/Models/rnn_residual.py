import torch
import torch.nn as nn

class ResidualRNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True)
        self.proj = nn.Identity() if input_dim == hidden_size else nn.Linear(input_dim, hidden_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out + self.proj(x)

class RNN_Residual(nn.Module):
    """Stacked residual vanilla RNN blocks."""
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(num_layers):
            layers.append(ResidualRNNLayer(d, hidden_size))
            d = hidden_size
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.RNN):
                for name, p in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(p)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return self.head(x[:, -1, :])
