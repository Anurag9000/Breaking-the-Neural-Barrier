import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_Vanilla(nn.Module):
    """
    Vanilla RNN (tanh) classifier.
    - Input: sequence of shape (B, T, D)
    - RNN: hidden size H, num_layers L
    - Head: take last hidden state -> Linear(H, num_classes)
    """
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='tanh',
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, num_classes)
        self._init()

    def _init(self):
        for name, p in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, T, D)
        out, h_n = self.rnn(x)  # out: (B, T, H); h_n: (L, B, H)
        last = out[:, -1, :]
        return self.head(last)
