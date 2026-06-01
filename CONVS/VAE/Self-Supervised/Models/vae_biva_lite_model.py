import torch
import torch.nn as nn
import torch.nn.functional as F

# BIVA-lite: bidirectional inference with top-down and bottom-up paths (simplified single pass)

class BottomUp(nn.Module):
    def __init__(self, in_ch=1, z_top=16, z_low=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(True),
        )
        self.fc = nn.Linear(128*4*4, 256)
        self.mu_top = nn.Linear(256, z_top); self.lv_top = nn.Linear(256, z_top)
        self.mu_low = nn.Linear(256, z_low); self.lv_low = nn.Linear(256, z_low)
    def forward(self,x):
        h=self.conv(x).view(x.size(0),-1)
        h=F.relu(self.fc(h))
        return (self.mu_top(h), self.lv_top(h)), (self.mu_low(h), self.lv_low(h))

class TopDownPrior(nn.Module):
    def __init__(self, z_top=16, z_low=16):
        super().__init__()
        self.mu_low = nn.Linear(z_top, z_low); self.lv_low = nn.Linear(z_top, z_low)
    def forward(self, zt):
        return self.mu_low(zt), self.lv_low(zt)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, z_low=16):
        super().__init__()
        self.fc = nn.Linear(z_low, 128*4*4)
        self.de = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(32,out_ch,4,2,1),
        )
    def forward(self, zl):
        return self.de(self.fc(zl).view(zl.size(0),128,4,4))

class BIVALite(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_top=16, z_low=16, recon='bce'):
        super().__init__()
        self.bu = BottomUp(in_ch, z_top, z_low)
        self.td = TopDownPrior(z_top, z_low)
        self.dec = Decoder(out_ch, z_low)
        self.recon = recon
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self,x):
        (mu_t,lv_t),(mu_l,lv_l) = self.bu(x)
        zt = self.reparam(mu_t, lv_t)
        p_mu_l, p_lv_l = self.td(zt)
        zl = self.reparam(mu_l, lv_l)
        x_logits = self.dec(zl)
        klt = -0.5*(1+lv_t - mu_t.pow(2)-lv_t.exp()).sum(1).mean()
        kll = 0.5*( ((mu_l-p_mu_l).pow(2)+lv_l.exp())/p_lv_l.exp() + p_lv_l - lv_l - 1 ).sum(1).mean()
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')/x.size(0)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum')/x.size(0)
        return rec + klt + kll
