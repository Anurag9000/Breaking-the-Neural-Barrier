import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqCLRGRU(nn.Module):
    """
    SimCLR-style contrastive learning for sequences with a single GRU encoder
    shared across both augmented views. Projection MLP maps to contrastive space.
    """
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        # x: (B,T,D)
        h, _ = self.encoder(x)
        # global average pooling over time -> (B,H)
        h = h.mean(dim=1)
        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        return z

    @staticmethod
    def nt_xent(z1, z2, temperature: float = 0.2):
        z = torch.cat([z1, z2], dim=0)  # (2B, P)
        sim = torch.matmul(z, z.t()) / temperature
        B = z1.size(0)
        mask = torch.eye(2*B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
        loss = torch.nn.functional.cross_entropy(sim, labels)
        return loss
