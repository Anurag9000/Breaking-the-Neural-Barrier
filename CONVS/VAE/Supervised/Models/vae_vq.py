from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# VQ-VAE (single-model, vector quantization with straight-through estimator)
# ------------------------------

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1/n_e, 1/n_e)

    def forward(self, z):
        # z: (B,C,H,W)
        B, C, H, W = z.shape
        flat = z.permute(0,2,3,1).contiguous().view(-1, C)  # (BHW,C)
        dist = (flat.pow(2).sum(1, keepdim=True) - 2*flat @ self.embedding.weight.t() + self.embedding.weight.pow(2).sum(1))
        idx = torch.argmin(dist, dim=1)
        quant = self.embedding(idx).view(B,H,W,C).permute(0,3,1,2).contiguous()
        # compute losses
        commit_loss = F.mse_loss(quant.detach(), z)
        codebook_loss = F.mse_loss(quant, z.detach())
        quant = z + (quant - z).detach()  # straight-through
        loss = commit_loss + self.beta * codebook_loss
        return quant, loss, idx

@dataclass
class VQVAEConfig:
    in_channels: int = 3
    width: int = 128
    n_embed: int = 512
    embed_dim: int = 64

class VQVAE(nn.Module):
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.enc = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(w, w, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(w, cfg.embed_dim, 3, 1, 1)
        )
        self.quant = VectorQuantizer(cfg.n_embed, cfg.embed_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(cfg.embed_dim, w, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(w, w//2, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(w//2, cfg.in_channels, 3, 1, 1)
        )

    def forward(self, x):
        z_e = self.enc(x)
        z_q, vq_loss, _ = self.quant(z_e)
        x_hat = torch.sigmoid(self.dec(z_q))
        return x_hat, vq_loss

    def loss(self, x, x_hat, vq_loss):
        recon = F.mse_loss(x_hat, x)
        return recon + vq_loss, recon.detach(), vq_loss.detach()
