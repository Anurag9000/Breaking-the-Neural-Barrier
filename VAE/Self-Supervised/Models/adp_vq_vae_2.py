from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQVAE2Config:
    in_channels: int = 3
    img_size: int = 32
    emb_dim_top: int = 64
    emb_dim_bottom: int = 64
    n_codes_top: int = 256
    n_codes_bottom: int = 512
    width: int = 128

class VQ(nn.Module):
    def __init__(self, K, D, decay=0.99, eps=1e-5):
        super().__init__()
        self.K=K; self.D=D; self.decay=decay; self.eps=eps
        self.emb = nn.Embedding(K, D)
        self.emb.weight.data.normal_()
        self.register_buffer('ema_cs', torch.zeros(K))
        self.register_buffer('ema_w', torch.randn(K, D))
    def forward(self, z_e):
        B,C,H,W = z_e.shape
        flat = z_e.permute(0,2,3,1).contiguous().view(-1, C)
        d = (flat.pow(2).sum(1,True) - 2*flat@self.emb.weight.t() + self.emb.weight.pow(2).sum(1))
        inds = d.argmin(1)
        z_q = self.emb(inds).view(B,H,W,C).permute(0,3,1,2).contiguous()
        oh = F.one_hot(inds, self.K).type_as(z_e)
        self.ema_cs.data.mul_(self.decay).add_(oh.sum(0), alpha=1-self.decay)
        self.ema_w.data.mul_(self.decay).add_(oh.t()@flat, alpha=1-self.decay)
        n = self.ema_cs.sum(); cs = ((self.ema_cs+self.eps)/(n+self.K*self.eps))*n
        w = self.ema_w / cs.unsqueeze(1)
        self.emb.weight.data.copy_(w)
        cb = F.mse_loss(z_e, z_q.detach())
        cm = F.mse_loss(z_e.detach(), z_q)
        z_q = z_e + (z_q - z_e).detach()
        return z_q, cb, cm

class TopEnc(nn.Module):
    def __init__(self, in_ch, w, emb_dim):
        super().__init__()
        ch=w
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(ch, ch, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(ch, emb_dim, 1)
        )
    def forward(self,x): return self.net(x)

class BottomEnc(nn.Module):
    def __init__(self, in_ch, w, emb_dim):
        super().__init__()
        ch=w
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(ch, emb_dim, 1)
        )
    def forward(self,x): return self.net(x)

class TopDec(nn.Module):
    def __init__(self, out_ch, w, emb_dim):
        super().__init__()
        ch=w
        self.net = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, ch, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(ch, out_ch, 4, 2, 1)
        )
    def forward(self,z): return self.net(z)

class BottomDec(nn.Module):
    def __init__(self, out_ch, w, emb_dim):
        super().__init__()
        ch=w
        self.net=nn.Sequential(
            nn.Conv2d(emb_dim*2, ch, 3, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(ch, out_ch, 1)
        )
    def forward(self, zb, zt_upsampled):
        h=torch.cat([zb, zt_upsampled], dim=1)
        return torch.sigmoid(self.net(h))

class VQVAE2(nn.Module):
    def __init__(self, cfg: VQVAE2Config):
        super().__init__()
        self.cfg=cfg
        self.enc_top = TopEnc(cfg.in_channels, cfg.width, cfg.emb_dim_top)
        self.enc_bottom = BottomEnc(cfg.in_channels, cfg.width, cfg.emb_dim_bottom)
        self.vq_top = VQ(cfg.n_codes_top, cfg.emb_dim_top)
        self.vq_bottom = VQ(cfg.n_codes_bottom, cfg.emb_dim_bottom)
        self.dec_top = TopDec(cfg.emb_dim_top, cfg.width, cfg.emb_dim_top)
        self.dec_bottom = BottomDec(cfg.in_channels, cfg.width, cfg.emb_dim_bottom)

    def forward(self,x):
        zt_e = self.enc_top(x)          # (B,Dt,8,8)
        zb_e = self.enc_bottom(x)       # (B,Db,16,16)
        zt_q, cb_t, cm_t = self.vq_top(zt_e)
        zt_up = F.interpolate(zt_q, scale_factor=2, mode='nearest')  # (B,Dt,16,16)
        zb_q, cb_b, cm_b = self.vq_bottom(zb_e)
        top_dec = self.dec_top(zt_q)    # (B,Dt,32,32)
        xr = self.dec_bottom(zb_q, zt_up)
        loss_cb = cb_t + cb_b
        loss_cm = cm_t + cm_b
        return xr, loss_cb, loss_cm

    def loss_fn(self, x, xr, cb, cm) -> Dict[str, torch.Tensor]:
        recon = F.binary_cross_entropy(xr, x, reduction='mean')
        loss = recon + cb + 0.25 * cm
        return {"loss":loss, "recon":recon, "codebook":cb, "commit":cm}
