import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-group categorical latent VAE (Gumbel-Softmax), K groups of C categories

class Encoder(nn.Module):
    def __init__(self, in_ch=1, K=4, C=16):
        super().__init__()
        self.K, self.C = K, C
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(True),
        )
        self.fc = nn.Linear(128*4*4, K*C)
    def forward(self,x):
        h=self.conv(x).view(x.size(0),-1)
        return self.fc(h).view(x.size(0), self.K, self.C)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, K=4, C=16):
        super().__init__()
        self.fc = nn.Linear(K*C, 128*4*4)
        self.de = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(32,out_ch,4,2,1),
        )
    def forward(self, y):
        return self.de(self.fc(y).view(y.size(0),128,4,4))

class MultiCatGumbelVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, K=4, C=16, tau=1.0, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, K, C)
        self.dec = Decoder(out_ch, K, C)
        self.K, self.C, self.tau = K, C, tau
        self.recon = recon
    @staticmethod
    def gumbel_softmax_sample(logits, tau):
        g = -torch.log(-torch.log(torch.rand_like(logits)+1e-8)+1e-8)
        y = torch.softmax((logits+g)/tau, dim=-1)
        return y
    def forward(self,x):
        logits = self.enc(x)  # [B,K,C]
        y = self.gumbel_softmax_sample(logits, self.tau).flatten(1)  # [B,K*C]
        x_logits = self.dec(y)
        q = torch.softmax(logits, dim=-1)
        log_q = torch.log(q+1e-8)
        kl = (q*(log_q - torch.log(torch.tensor(1.0/self.C, device=x.device)))).sum((-1,-2)).mean()
        if self.recon=='bce':
            rec=F.binary_cross_entropy_with_logits(x_logits,x,reduction='sum')/x.size(0)
        else:
            rec=F.mse_loss(torch.sigmoid(x_logits),x,reduction='sum')/x.size(0)
        return rec + kl
