import torch
import torch.nn as nn
import torch.nn.functional as F

# IWAE (Importance Weighted Autoencoder) with K samples

class Encoder(nn.Module):
    def __init__(self, in_ch=1, z_dim=32):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(True),
        )
        self.mu=nn.Linear(128*4*4,z_dim)
        self.lv=nn.Linear(128*4*4,z_dim)
    def forward(self,x):
        h=self.conv(x).view(x.size(0),-1)
        return self.mu(h), self.lv(h)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, z_dim=32):
        super().__init__()
        self.fc=nn.Linear(z_dim,128*4*4)
        self.de=nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(32,out_ch,4,2,1)
        )
    def forward(self,z):
        return self.de(self.fc(z).view(z.size(0),128,4,4))

class IWAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, K=5, recon='bce'):
        super().__init__()
        self.enc=Encoder(in_ch,z_dim)
        self.dec=Decoder(out_ch,z_dim)
        self.K=K; self.recon=recon
    def forward(self,x):
        B=x.size(0); K=self.K
        mu,lv=self.enc(x)
        std=(0.5*lv).exp()
        eps=torch.randn(K,B,mu.size(1), device=x.device)
        z = mu.unsqueeze(0)+std.unsqueeze(0)*eps  # [K,B,D]
        zf=z.view(K*B,-1)
        x_logits=self.dec(zf).view(K,B,*x.shape[1:])
        if self.recon=='bce':
            log_px_z = -F.binary_cross_entropy_with_logits(x_logits, x.unsqueeze(0).expand_as(x_logits), reduction='none').flatten(2).sum(-1)  # [K,B]
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x.unsqueeze(0).expand_as(x_logits), reduction='none').flatten(2).sum(-1)
            log_px_z = -rec
        log_pz = -0.5*(z.pow(2) + torch.log(torch.tensor(2*3.141592653589793, device=x.device))).sum(-1)  # [K,B]
        log_qz_x = -0.5*(((z - mu.unsqueeze(0))**2)/(std.unsqueeze(0)**2) + 2*torch.log(std.unsqueeze(0)) + torch.log(torch.tensor(2*3.141592653589793, device=x.device))).sum(-1)
        log_w = log_px_z + log_pz - log_qz_x  # [K,B]
        # IWAE objective = - E_B[ log (1/K sum_k exp(log_w)) ]
        log_mean_w = torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(float(K), device=x.device))
        return -(log_mean_w.mean())
