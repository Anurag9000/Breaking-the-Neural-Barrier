import torch
import torch.nn as nn
from typing import Optional

class MaskedTimeModel(nn.Module):
    """
    Bidirectional GRU encoder with a linear head to predict masked time-steps.
    - Input: (B, T, D)
    - mask: boolean mask (B, T), True where tokens are masked and must be predicted
    - Loss computed in runner: only on masked positions
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2*hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        return self.head(h)

if __name__ == '__main__':
    B,T,D=2,8,4
    net = MaskedTimeModel(D, 16)
    x = torch.randn(B,T,D)
    y = net(x)
    print(y.shape)
