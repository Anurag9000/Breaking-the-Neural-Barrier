import torch
import torch.nn as nn
import torch.nn.functional as F

# Ladder VAE (2-level) with top-down refinement

class BottomUp(nn.Module):
    def __init__(self, in_ch=1, z_high=16, z_low=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,3,2,1), nn.ReLU(True)
        )
        self.fc_h = nn.Linear(128*4*4, 128)
        self.mu_high = nn.Linear(128, z_high); self.lv_high = nn.Linear(128, z_high)
        self.mu_low_bu = nn.Linear(128, z_low); self.lv_low_bu = nn.Linear(128, z_low)
    def forward(self,x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1).view(x.size(0), -1)
        h = F.relu(self.fc_h(h2))
        return (self.mu_high(h), self.lv_high(h)), (self.mu_low_bu(h), self.lv_low_bu(h))

class TopDown(nn.Module):
    def __init__(self, z_high=16, z_low=32):
        super().__init__()
        self.mu_low_td = nn.Linear(z_high, z_low); self.lv_low_td = nn.Linear(z_high, z_low)
    def forward(self, zh):
        return self.mu_low_td(zh), self.lv_low_td(zh)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, z_low=32):
        super().__init__()
        self.fc = nn.Linear(z_low, 128*4*4)
        self.de = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(32,out_ch,4,2,1)
        )
    def forward(self, zl):
        return self.de(self.fc(zl).view(zl.size(0),128,4,4))

class LadderVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_high=16, z_low=32, recon='bce'):
        super().__init__()
        self.bu = BottomUp(in_ch, z_high, z_low)
        self.td = TopDown(z_high, z_low)
        self.dec = Decoder(out_ch, z_low)
        self.recon = recon
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self,x):
        (mu_h, lv_h), (mu_l_bu, lv_l_bu) = self.bu(x)
        zh = self.reparam(mu_h, lv_h)
        mu_l_td, lv_l_td = self.td(zh)
        # precision-weighted fusion of low-level stats
        prec_bu = (lv_l_bu.exp()).reciprocal()
        prec_td = (lv_l_td.exp()).reciprocal()
        mu_l = (mu_l_bu*prec_bu + mu_l_td*prec_td)/(prec_bu+prec_td)
        lv_l = torch.log((prec_bu+prec_td).reciprocal())
        zl = self.reparam(mu_l, lv_l)
        x_logits = self.dec(zl)
        kl_h = -0.5*(1+lv_h - mu_h.pow(2)-lv_h.exp()).sum(1).mean()
        kl_l = 0.5*( ((mu_l - mu_l_td).pow(2) + lv_l.exp())/lv_l_td.exp() + lv_l_td - lv_l - 1 ).sum(1).mean()
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')/x.size(0)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum')/x.size(0)
        return rec + kl_h + kl_l
