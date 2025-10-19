import torch
import torch.nn as nn

class RNN_Orthogonal(nn.Module):
    """Vanilla RNN (tanh) with orthogonal recurrent matrices."""
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.rnns = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        d = input_dim
        for _ in range(num_layers):
            r = nn.RNN(d, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True)
            # Orthogonalize recurrent weight
            for name, p in r.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(p)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(p)
                elif 'bias' in name:
                    nn.init.zeros_(p)
            self.rnns.append(r)
            d = hidden_size
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        for r in self.rnns:
            x, _ = r(x)
            x = self.drop(x)
        return self.head(x[:, -1, :])
