import torch
import torch.nn as nn

class ARGRU(nn.Module):
    """
    Autoregressive next-step prediction with GRU backbone.
    Teacher forcing handled in runner: input sequence x[:, :-1] predicts x[:, 1:].
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        assert not bidirectional, "AR model uses causal direction; set bidirectional=False"
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):  # x: (B, T-1, D)
        h, _ = self.gru(x)
        return self.head(h)  # (B, T-1, D)

class ARRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, nonlinearity='tanh', batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        h,_ = self.rnn(x)
        return self.head(h)
