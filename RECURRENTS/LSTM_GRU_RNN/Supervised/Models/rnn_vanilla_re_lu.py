import torch
import torch.nn as nn

class RNN_Vanilla_ReLU(nn.Module):
    """Vanilla RNN with ReLU nonlinearity."""
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='relu',
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, num_classes)
        self._init()

    def _init(self):
        for name, p in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(p, a=0)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])
