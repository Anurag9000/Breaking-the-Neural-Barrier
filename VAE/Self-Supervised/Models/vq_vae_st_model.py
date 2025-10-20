import torch
import torch.nn as nn
import torch.nn.functional as F

# VQ-VAE (Straight-Through, no EMA)

class Encoder(nn.Module):
    def __init__(self, in_ch=1, hid=128, z_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(hid, hid, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(hid, z_ch, 3, 1, 1)
        )
    def forward(self,x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, hid=128, z_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_ch, hid, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(hid, hid, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(hid, out_ch, 3, 1, 1)
        )
    def forward(self,z_q):
        return self.net(z_q)

class VectorQuantizer(nn.Module):
    def __init__(self, K=512, D=64, beta=0.25):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(K, D))
        self.beta = beta
    def forward(self, z):
        # z: [B, D, H, W] -> flatten to [BHW, D]
        B,D,H,W = z.shape
        z_flat = z.permute(0,2,3,1).contiguous().view(-1, D)
        # distances to codebook
        cb = self.codebook  # [K,D]
        d2 = (z_flat.pow(2).sum(1, keepdim=True) + cb.pow(2).sum(1).unsqueeze(0) - 2*z_flat@cb.t())
        idx = torch.argmin(d2, dim=1)  # [BHW]
        z_q = cb[idx].view(B,H,W,D).permute(0,3,1,2).contiguous()  # [B,D,H,W]
        # losses
        commit = F.mse_loss(z.detach(), z_q)
        codebk = F.mse_loss(z, z_q.detach())
        loss_vq = codebk + self.beta*commit
        # straight-through
        z_q_st = z + (z_q - z).detach()
        return z_q_st, idx.view(B,1,H,W), loss_vq

class VQVAE_ST(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, hid=128, z_ch=64, K=512, beta=0.25, recon='bce'):
        super().__init__()
        self.enc=Encoder(in_ch,hid,z_ch)
        self.dec=Decoder(out_ch,hid,z_ch)
        self.vq=VectorQuantizer(K, z_ch, beta)
        self.recon=recon
    def forward(self,x):
        z_e = self.enc(x)
        z_q, _, loss_vq = self.vq(z_e)
        x_logits = self.dec(z_q)
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')/x.size(0)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum')/x.size(0)
        return rec + loss_vq
