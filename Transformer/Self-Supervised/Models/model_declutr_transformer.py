import torch
import torch.nn as nn
import torch.nn.functional as F
from model_simcse_transformer import TransformerEncoder, SimpleTokenizer

class DeCLUTRTransformer(nn.Module):
    """DeCLUTR: contrastive sentence learning with span-based views (single encoder)."""
    def __init__(self, vocab, dim=256, depth=4, heads=8, mlp_ratio=4.0, proj_dim=128):
        super().__init__()
        self.encoder = TransformerEncoder(vocab, dim, depth, heads, mlp_ratio)
        self.proj = nn.Linear(dim, proj_dim)
    def forward(self, x1, x2, temperature=0.05):
        z1 = F.normalize(self.proj(self.encoder(x1)), dim=-1)
        z2 = F.normalize(self.proj(self.encoder(x2)), dim=-1)
        z = torch.cat([z1,z2], dim=0)
        sim = (z @ z.t()) / temperature
        B = z1.size(0)
        mask = torch.eye(2*B, device=z.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        targets = torch.cat([torch.arange(B,2*B), torch.arange(0,B)]).to(z.device)
        return F.cross_entropy(sim, targets)
