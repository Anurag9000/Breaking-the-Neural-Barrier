import torch
import torch.nn as nn
import torch.nn.functional as F

# PixelVAE+ : deeper masked conv stack with gated activations

class GatedMaskedConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch, k, mask_type='B'):
        super().__init__(in_ch, 2*out_ch, k, padding=k//2)
        self.out_ch = out_ch
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, kh, kw = self.weight.shape
        yc, xc = kh//2, kw//2
        self.mask[:,:,yc+1:,:] = 0
        self.mask[:,:,yc,:xc + (0 if mask_type=='A' else 1)] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        h = super().forward(x)
        a, b = h.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)

class Encoder(nn.Module):
    def __init__(self, in_ch=1, z_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(True),
        )
        self.mu = nn.Linear(128*4*4, z_dim)
        self.lv = nn.Linear(128*4*4, z_dim)
    def forward(self,x):
        h=self.conv(x).view(x.size(0),-1)
        return self.mu(h), self.lv(h)

class PixelDecoderPlus(nn.Module):
    def __init__(self, out_ch=1, z_dim=32, hidden=64, depth=6):
        super().__init__()
        self.fc = nn.Linear(z_dim, hidden*28*28)
        layers = []
        for i in range(depth):
            layers += [GatedMaskedConv2d(hidden, hidden, 3, 'B')]
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv2d(hidden, out_ch, 1)
    def forward(self, z):
        h = self.fc(z).view(z.size(0), -1, 28, 28)
        h = self.net(h)
        return self.out(h)

class PixelVAEPlus(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, recon='bce', depth=6):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = PixelDecoderPlus(out_ch, z_dim, depth=depth)
        self.recon = recon
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self,x):
        mu,lv=self.enc(x)
        z=self.reparam(mu,lv)
        x_logits=self.dec(z)
        kl=-0.5*(1+lv - mu.pow(2)-lv.exp()).sum(1).mean()
        if self.recon=='bce':
            rec=F.binary_cross_entropy_with_logits(x_logits,x,reduction='sum')/x.size(0)
        else:
            rec=F.mse_loss(torch.sigmoid(x_logits),x,reduction='sum')/x.size(0)
        return rec + kl
