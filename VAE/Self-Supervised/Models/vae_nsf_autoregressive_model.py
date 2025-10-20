import torch
import torch.nn as nn
import torch.nn.functional as F

# Neural Spline Flow (Rational-Quadratic) autoregressive posterior VAE
# Conditioner uses a lightweight MADE-style MLP (simplified) to output per-dimension spline params

# ---- RQS util (elementwise) ----

def rqs_1d(x, widths, heights, derivatives, B=3.0):
    K = widths.size(-1)
    widths = widths/widths.sum(-1, keepdim=True)
    heights = heights/heights.sum(-1, keepdim=True)
    derivatives = torch.clamp(derivatives, 1e-3, 1e3)
    cw = torch.cumsum(widths, dim=-1)
    ch = torch.cumsum(heights, dim=-1)
    cw = F.pad(cw, (1,0), value=0.0)
    ch = F.pad(ch, (1,0), value=0.0)
    x_s = (x + B)/(2*B)
    x_s = x_s.clamp(0.0, 1.0)
    # bin index
    k = torch.sum(x_s.unsqueeze(-1) >= cw, dim=-1) - 1
    k = k.clamp(0, K-1)
    w = widths.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    h = heights.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    cwi = cw.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    chi = ch.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    dl = derivatives.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    dr = derivatives.gather(-1, (k+1).unsqueeze(-1)).squeeze(-1)
    t = (x_s - cwi)/(w + 1e-12)
    a = h; b = dl; c = dr
    y_unit = chi + a*t + (t*(1-t))*(a*(1-a)*(2*t - 1) + (b*(1 - t) + c*t) - a)
    # derivative wrt x
    dy_dt = a + (1-2*t)*(a*(1-a)*(2*t-1) + (b*(1-t)+c*t) - a) + t*(1-t)*(2*a*(1-a) + (c-b))
    dy_dx = dy_dt/(w + 1e-12)/(2*B)
    y = y_unit*(2*B) - B
    logdet = torch.log(torch.abs(dy_dx) + 1e-12)
    return y, logdet

class MADELite(nn.Module):
    def __init__(self, dim, hidden=256, K=8):
        super().__init__()
        self.dim = dim; self.K = K
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, dim*(2*K + (K+1)))
        )
    def forward(self, z):
        out = self.net(z)
        K = self.K; D = z.size(1)
        widths, heights, derivatives = torch.split(out, [D*K, D*K, D*(K+1)], dim=-1)
        widths = widths.view(-1, D, K).softplus() + 1e-3
        heights = heights.view(-1, D, K).softplus() + 1e-3
        derivatives = derivatives.view(-1, D, K+1).softplus() + 1e-3
        return widths, heights, derivatives

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

class Decoder(nn.Module):
    def __init__(self, out_ch=1, z_dim=32):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128*4*4)
        self.de = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(32,out_ch,4,2,1)
        )
    def forward(self, z):
        return self.de(self.fc(z).view(z.size(0),128,4,4))

class VAE_NSF_AR(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, n_flows=4, K=8, B=3.0, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
        self.recon = recon
        self.K = K; self.B = B
        self.conditioners = nn.ModuleList([MADELite(z_dim, hidden=256, K=K) for _ in range(n_flows)])
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self, x):
        mu, lv = self.enc(x)
        z = self.reparam(mu, lv)
        log_q0 = -0.5*((z-mu).pow(2)/lv.exp() + lv + torch.log(torch.tensor(2*3.141592653589793, device=x.device))).sum(-1)
        sum_logdet = torch.zeros(x.size(0), device=x.device)
        z_k = z
        for cond in self.conditioners:
            widths, heights, derivatives = cond(z_k)
            # elementwise transform per dimension
            y_list = []
            ld_list = []
            for d in range(z_k.size(1)):
                y_d, ld_d = rqs_1d(z_k[:, d], widths[:, d], heights[:, d], derivatives[:, d], B=self.B)
                y_list.append(y_d.unsqueeze(1)); ld_list.append(ld_d)
            z_k = torch.cat(y_list, dim=1)
            sum_logdet += torch.stack(ld_list, dim=1).sum(-1)
        x_logits = self.dec(z_k)
        log_pz = -0.5*(z_k.pow(2) + torch.log(torch.tensor(2*3.141592653589793, device=x.device))).sum(-1)
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='none').flatten(1).sum(-1)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='none').flatten(1).sum(-1)
        elbo = -rec + log_pz - (log_q0 - sum_logdet)
        return (-elbo).mean()
