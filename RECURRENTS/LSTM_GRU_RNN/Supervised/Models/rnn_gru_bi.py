import torch
import torch.nn as nn

class RNN_GRU_Bi(nn.Module):
    """Bidirectional standard GRU classifier (single model)."""
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_size*2, num_classes)
        self._init()

    def _init(self):
        for name, p in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])
