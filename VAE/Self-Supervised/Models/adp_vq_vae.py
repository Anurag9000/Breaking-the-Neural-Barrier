from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 64   # embedding dimension
    n_codes: int = 512     # codebook size
    width: int = 128
    depth: int = 2
    beta_commit: float = 0.25


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

class Encoder(nn.Module):
    def __init__(self,in_ch,w,d,emb_dim):
        super().__init__()
        ch=w; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        layers+=[nn.Conv2d(ch, emb_dim, 1)]  # produce embedding grid
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self,out_ch,w,d,emb_dim):
        super().__init__()
        ch=w
        layers=[nn.Conv2d(emb_dim, ch, 3, 1, 1), nn.ReLU(True)]
        for _ in range(d-1):
            layers+=[nn.ConvTranspose2d(ch,ch,4,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        layers+=[nn.ConvTranspose2d(ch, out_ch, 4, 2, 1)]
        self.net=nn.Sequential(*layers)
    def forward(self,zq): return torch.sigmoid(self.net(zq))

class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_codes:int, emb_dim:int, decay:float=0.99, eps:float=1e-5):
        super().__init__()
        self.n_codes=n_codes; self.emb_dim=emb_dim; self.decay=decay; self.eps=eps
        self.embedding = nn.Embedding(n_codes, emb_dim)
        self.embedding.weight.data.normal_()
        self.register_buffer('ema_cluster_size', torch.zeros(n_codes))
        self.register_buffer('ema_weight', torch.randn(n_codes, emb_dim))

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: (B, C=emb_dim, H, W)
        B,C,H,W = z_e.shape
        flat = z_e.permute(0,2,3,1).contiguous().view(-1, C)  # (BHW, C)
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))  # (BHW, K)
        inds = torch.argmin(d, dim=1)  # (BHW,)
        z_q = self.embedding(inds).view(B, H, W, C).permute(0,3,1,2).contiguous()

        # EMA updates
        one_hot = F.one_hot(inds, self.n_codes).type_as(z_e)
        new_cluster_size = one_hot.sum(0)
        self.ema_cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1-self.decay)
        dw = one_hot.t() @ flat  # (K,C)
        self.ema_weight.data.mul_(self.decay).add_(dw, alpha=1-self.decay)
        n = self.ema_cluster_size.sum()
        cluster_size = ((self.ema_cluster_size + self.eps) / (n + self.n_codes * self.eps)) * n
        embed_normalized = self.ema_weight / cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)

        # losses
        commit_loss = F.mse_loss(z_e.detach(), z_q)
        codebook_loss = F.mse_loss(z_e, z_q.detach())
        z_q = z_e + (z_q - z_e).detach()  # straight-through
        return z_q, codebook_loss, commit_loss


class VQVAE(nn.Module):
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=Encoder(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.quant = VectorQuantizerEMA(cfg.n_codes, cfg.latent_dim)
        self.dec=Decoder(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)

    def forward(self,x):
        z_e=self.enc(x)
        z_q, cb_loss, commit_loss = self.quant(z_e)
        xr=self.dec(z_q)
        return xr, cb_loss, commit_loss

    def loss_fn(self, x, xr, cb_loss, commit_loss) -> Dict[str, torch.Tensor]:
        recon=F.binary_cross_entropy(xr, x, reduction='mean')
        loss = recon + cb_loss + self.cfg.beta_commit * commit_loss
        return {"loss":loss, "recon":recon, "codebook":cb_loss, "commit":commit_loss}
