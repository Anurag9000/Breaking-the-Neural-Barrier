import torch
import torch.nn as nn
import torch.nn.functional as F

class BarlowGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        h,_ = self.encoder(x)
        h = h.mean(dim=1)
        z = self.proj(h)
        z = (z - z.mean(0)) / (z.std(0) + 1e-6)
        return z

    @staticmethod
    def barlow_loss(z1, z2, lambd: float = 5e-3):
        N, D = z1.shape
        c = (z1.T @ z2) / N  # cross-correlation D x D
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c - torch.diag(torch.diag(c))).pow_(2).sum()
        return on_diag + lambd * off_diag

if __name__ == '__main__':
    B,T,D=8,64,16
    net = BarlowGRU(D, 64, 32)
    x = torch.randn(B,T,D)
    z = net(x)
    print(z.shape)
