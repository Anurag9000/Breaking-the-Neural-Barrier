import torch
import torch.nn as nn

class SBDGRU(nn.Module):
    """Predict boundary token per time-step (binary segmentation)."""
    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int=1):
        super().__init__()
        self.encoder=nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head=nn.Linear(2*hidden_dim, 1)
    def forward(self, x):
        h,_=self.encoder(x)
        return self.head(h).squeeze(-1)  # (B,T)
