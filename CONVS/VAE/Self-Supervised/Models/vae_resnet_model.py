import torch
import torch.nn as nn
import torch.nn.functional as F

# ResNet-style VAE encoder/decoder blocks

class ResBlockEnc(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        self.skip = nn.Conv2d(c_in, c_out, 1, stride, 0) if (stride!=1 or c_in!=c_out) else nn.Identity()
    def forward(self,x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(h + self.skip(x))

class ResBlockDec(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(c_in, c_out, 4, 2, 1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        self.skip = nn.ConvTranspose2d(c_in, c_out, 4, 2, 1)
    def forward(self,x):
        h = F.relu(self.deconv1(x))
        h = self.conv2(h)
        return F.relu(h + self.skip(x))

class Encoder(nn.Module):
    def __init__(self, in_ch=1, z_dim=32):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, 32, 3, 1, 1)
        self.b1 = ResBlockEnc(32, 64, stride=2)
        self.b2 = ResBlockEnc(64, 128, stride=2)
        self.b3 = ResBlockEnc(128, 128, stride=2)
        self.mu = nn.Linear(128*4*4, z_dim)
        self.lv = nn.Linear(128*4*4, z_dim)
    def forward(self,x):
        h = F.relu(self.stem(x))
        h = self.b1(h); h = self.b2(h); h = self.b3(h)
        h = h.view(x.size(0), -1)
        return self.mu(h), self.lv(h)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, z_dim=32):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128*4*4)
        self.b1 = ResBlockDec(128, 128)
        self.b2 = ResBlockDec(128, 64)
        self.b3 = ResBlockDec(64, 32)
        self.out = nn.Conv2d(32, out_ch, 3, 1, 1)
    def forward(self,z):
        h = self.fc(z).view(z.size(0),128,4,4)
        h = self.b1(h); h = self.b2(h); h = self.b3(h)
        return self.out(h)

class ResNetVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
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
        return rec+kl
