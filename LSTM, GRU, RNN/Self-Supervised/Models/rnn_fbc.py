import torch
import torch.nn as nn

class FBCGRU(nn.Module):
    """
    Binary classifier: predict whether a sequence is forward (label 1) or reversed (label 0).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        h,_ = self.encoder(x)
        h = h.mean(dim=1)
        logit = self.head(h)
        return logit.squeeze(-1)

if __name__ == '__main__':
    B,T,D=4,32,8
    net = FBCGRU(D, 64)
    x = torch.randn(B,T,D)
    y = net(x)
    print(y.shape)
