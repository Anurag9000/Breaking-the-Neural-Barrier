import torch
import torch.nn as nn

class APCGRU(nn.Module):
    """
    Autoregressive Predictive Coding with GRU backbone.
    Predicts K-steps ahead features using regression on hidden states.
    """
    def __init__(self, input_dim: int, hidden_dim: int, K: int = 3, num_layers: int = 1):
        super().__init__()
        self.K = K
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (B,T,D)
        h, _ = self.encoder(x)
        # predict x_{t+K} from h_t
        yhat = self.head(h)
        return yhat  # (B,T,D); runner will align target at t+K

if __name__ == '__main__':
    B,T,D=2,16,8
    net = APCGRU(D, 64, K=3)
    x = torch.randn(B,T,D)
    yhat = net(x)
    print(yhat.shape)
