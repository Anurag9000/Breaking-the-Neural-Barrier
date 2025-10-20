from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PixelVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 32
    width: int = 64
    depth: int = 2

# Minimal masked convolution (PixelCNN-like) for autoregressive decoder
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type: str, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert mask_type in ['A','B']
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, kH, kW = self.weight.size()
        yc, xc = kH//2, kW//2
        self.mask[:,:,yc+1:] = 0
        self.mask[:,:,yc, xc + (1 if mask_type=='A' else 0):] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class Enc(nn.Module):
    def __init__(self,in_ch,w,d,latent):
        super().__init__()
        ch=w; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.net=nn.Sequential(*layers)
        self.mu=nn.Linear(ch*(32//(2**d))*(32//(2**d)), latent)
        self.lv=nn.Linear(ch*(32//(2**d))*(32//(2**d)), latent)
        self.ch=ch; self.fs=32//(2**d)
    def forward(self,x):
        h=self.net(x).flatten(1)
        return self.mu(h), self.lv(h)

class PixelDecoder(nn.Module):
    def __init__(self, out_ch, w, d, latent):
        super().__init__()
        # map z to spatial feature map
        fs=32//(2**d); ch=w
        self.fc = nn.Linear(latent, ch*fs*fs)
        ups = []
        for _ in range(d-1): ups += [nn.ConvTranspose2d(ch,ch,4,2,1), nn.ReLU(True)]
        ups += [nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.ReLU(True)]  # final up to 32x32
        self.ups = nn.Sequential(*ups)
        # autoregressive head
        self.pixelcnn = nn.Sequential(
            MaskedConv2d(ch, ch, 3, padding=1, mask_type='A'), nn.ReLU(True),
            MaskedConv2d(ch, ch, 3, padding=1, mask_type='B'), nn.ReLU(True),
            nn.Conv2d(ch, out_ch, 1)
        )
    def forward(self,z):
        fs=32//(2**2)  # default matches depth=2 in cfg; robust calc below if needed
        h=self.fc(z)
        # infer fs from channels
        ch = self.ups[0].in_channels if isinstance(self.ups[0], nn.ConvTranspose2d) else 64
        fs = int((h.numel()//(h.size(0)*ch))**0.5)
        h=h.view(h.size(0), ch, fs, fs)
        h=self.ups(h)
        return torch.sigmoid(self.pixelcnn(h))

class PixelVAE(nn.Module):
    def __init__(self, cfg: PixelVAEConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=Enc(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.dec=PixelDecoder(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)

    @staticmethod
    def reparameterize(mu, lv):
        std=(0.5*lv).exp(); return mu+torch.randn_like(std)*std

    def forward(self,x):
        mu,lv=self.enc(x); z=self.reparameterize(mu,lv); xr=self.dec(z)
        return xr, mu, lv

    def loss_fn(self,x,xr,mu,lv)->Dict[str,torch.Tensor]:
        recon=F.binary_cross_entropy(xr,x,reduction='mean')
        kl=-0.5*torch.sum(1+lv-mu.pow(2)-lv.exp(),dim=1).mean()
        return {"loss":recon+kl, "recon":recon, "kl":kl}
