import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE with RealNVP-style flow decoder: p(x|z) modeled via flow g(u; z) with base u~N(0, I)

class AffineCoupling(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + cond_dim, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, channels*2, 1, 1, 0)
        )
        self.mask = None
    def set_mask(self, mask):
        self.mask = mask
    def forward(self, x, c):
        x_a = x * self.mask
        c_map = c.unsqueeze(-1).unsqueeze(-1).expand(-1, c.size(1), x.size(2), x.size(3))
        h = self.net(torch.cat([x_a, c_map], dim=1))
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)
        y = x_a + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = ((1 - self.mask) * s).flatten(1).sum(-1)
        return y, log_det

class FlowDecoder(nn.Module):
    def __init__(self, z_dim=32, img_ch=1, steps=4):
        super().__init__()
        self.z_to_c = nn.Linear(z_dim, 16)
        self.steps = nn.ModuleList()
        for i in range(steps):
            ac = AffineCoupling(img_ch, cond_dim=16)
            mask = torch.zeros(1, img_ch, 1, 1)
            mask[:, :, :, :] = 1 if i % 2 == 0 else 0
            ac.set_mask(mask)
            self.steps.append(ac)
    def forward(self, z, x):
        # base u ~ N(0,1), transform to x; here compute log p(x|z)
        c = torch.relu(self.z_to_c(z))
        u = x
        log_det_total = torch.zeros(x.size(0), device=x.device)
        for ac in self.steps:
            u, ld = ac(u, c)
            log_det_total += ld
        # base log prob
        log_base = -0.5*(u.flatten(1).pow(2).sum(-1) + u[0].numel()*torch.log(torch.tensor(2*3.141592653589793, device=x.device)))
        return log_base + log_det_total

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
    def forward(self, x):
        h=self.conv(x).view(x.size(0),-1)
        return self.mu(h), self.lv(h)

class FlowDecoderVAE(nn.Module):
    def __init__(self, in_ch=1, z_dim=32):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.flowdec = FlowDecoder(z_dim, img_ch=in_ch)
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self, x):
        mu, lv = self.enc(x)
        z = self.reparam(mu, lv)
        log_px_given_z = self.flowdec(z, x)
        kl = -0.5*(1+lv - mu.pow(2) - lv.exp()).sum(1)
        loss = (-(log_px_given_z) + kl).mean()
        return loss
