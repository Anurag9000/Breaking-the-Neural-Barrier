import torch
import torch.nn as nn
import torch.nn.functional as F

# Spatial Broadcast Decoder VAE (SBD-VAE): broadcast z to (H,W) with XY coords

class Encoder(nn.Module):
    def __init__(self, in_ch=1, z_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,32,4,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(True),
            nn.Conv2d(64,128,4,2,1), nn.ReLU(True),
        )
        self.mu = nn.Linear(128*3*3, z_dim)
        self.lv = nn.Linear(128*3*3, z_dim)
    def forward(self,x):
        h=self.conv(x).view(x.size(0),-1)
        return self.mu(h), self.lv(h)

class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, out_ch=1, z_dim=16, H=24, W=24):
        super().__init__()
        self.H, self.W = H, W
        self.net = nn.Sequential(
            nn.Conv2d(z_dim+2,64,3,1,1), nn.ReLU(True),
            nn.Conv2d(64,64,3,1,1), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64,32,3,1,1), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32,out_ch,3,1,1)
        )
    def forward(self, z):
        B, D = z.size()
        H, W = self.H, self.W
        # coordinate grid in [-1,1]
        ys = torch.linspace(-1,1,H, device=z.device)
        xs = torch.linspace(-1,1,W, device=z.device)
        Y, X = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([X, Y], dim=0).unsqueeze(0).expand(B, -1, H, W)
        z_map = z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        inp = torch.cat([z_map, coords], dim=1)
        return self.net(inp)

class SBDVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=16, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = SpatialBroadcastDecoder(out_ch, z_dim, H=7, W=7)
        self.recon = recon
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self,x):
        mu, lv = self.enc(x)
        z = self.reparam(mu, lv)
        x_logits = self.dec(z)
        kl = -0.5*(1+lv - mu.pow(2)-lv.exp()).sum(1).mean()
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')/x.size(0)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum')/x.size(0)
        return rec + kl
