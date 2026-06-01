import torch
import torch.nn as nn
import torch.nn.functional as F
from model_simcse_transformer import TransformerEncoder, SimpleTokenizer

class VICRegTextTransformer(nn.Module):
    def __init__(self, vocab, dim=256, depth=4, heads=8, mlp_ratio=4.0,
                 proj_hidden=2048, proj_out=2048, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4):
        super().__init__()
        self.encoder = TransformerEncoder(vocab, dim, depth, heads, mlp_ratio)
        self.proj = nn.Sequential(
            nn.Linear(dim, proj_hidden), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_hidden), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out)
        )
        self.sim_coeff = sim_coeff; self.var_coeff = var_coeff; self.cov_coeff = cov_coeff; self.eps = eps

    def invariance(self, z1, z2):
        return (z1 - z2).pow(2).mean()

    def variance(self, z):
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return torch.mean(F.relu(1 - std))

    def covariance(self, z):
        z = z - z.mean(dim=0)
        N, D = z.shape
        c = (z.T @ z) / (N - 1)
        off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum() / D
        return off_diag

    def forward(self, x1, x2):
        h1 = self.encoder(x1); h2 = self.encoder(x2)
        z1 = self.proj(h1); z2 = self.proj(h2)
        inv = self.invariance(z1, z2)
        var = self.variance(z1) + self.variance(z2)
        cov = self.covariance(z1) + self.covariance(z2)
        return self.sim_coeff * inv + self.var_coeff * var + self.cov_coeff * cov
