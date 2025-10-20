import torch
import torch.nn as nn
import torch.nn.functional as F

# Neural Spline Flow (Rational-Quadratic Spline) coupling-flow posterior VAE
# Based on Durkan et al. 2019 (Neural Spline Flows) — single-model, posterior flows only

# ---- RQS utility ----
# Bound the transform to [-B, B] with linear tails, K bins

def rational_quadratic_spline(x, widths, heights, derivatives, B=3.0):
    # x: [..., D_t]
    # widths/heights: [..., D_t, K] (positive, sum to 1)
    # derivatives: [..., D_t, K+1] (positive)
    K = widths.size(-1)
    # normalize
    widths = widths / widths.sum(dim=-1, keepdim=True)
    heights = heights / heights.sum(dim=-1, keepdim=True)
    derivatives = torch.clamp(derivatives, 1e-3, 1e3)

    # CDF positions of knots
    cumwidths = torch.cumsum(widths, dim=-1)
    cumheights = torch.cumsum(heights, dim=-1)
    cumwidths = F.pad(cumwidths, (1,0), value=0.0)
    cumheights = F.pad(cumheights, (1,0), value=0.0)

    # scale to [-B, B]
    x_scaled = (x + B) / (2*B)

    # handle tails: outside [0,1] -> linear
    below = x_scaled <= 0
    above = x_scaled >= 1
    inside = (~below) & (~above)

    # locate bin index
    x_ins = x_scaled.masked_select(inside)
    if x_ins.numel() == 0:
        # all in tails
        y = x.clone()
        logdet = torch.zeros_like(x)
        # left tail slope = first derivative; right tail slope = last derivative
        # but for exact linear tails, slope = 1
        return y, logdet

    # for vectorized bin selection, build search via cumulative widths
    cumw = cumwidths[inside, :]
    # find bin k s.t. cumw[k] <= x < cumw[k+1]
    # use torch.bucketize on CPU-friendly way
    k = torch.bucketize(x_ins, cumw.transpose(0,1).contiguous().transpose(0,1)) - 1
    k = torch.clamp(k, 0, widths.size(-1)-1)

    # gather parameters for chosen bins
    w = widths[inside, :].gather(-1, k.unsqueeze(-1)).squeeze(-1)
    h = heights[inside, :].gather(-1, k.unsqueeze(-1)).squeeze(-1)
    cw = cumwidths[inside, :].gather(-1, k.unsqueeze(-1)).squeeze(-1)
    ch = cumheights[inside, :].gather(-1, k.unsqueeze(-1)).squeeze(-1)
    d_left = derivatives[inside, :].gather(-1, k.unsqueeze(-1)).squeeze(-1)
    d_right = derivatives[inside, :].gather(-1, k.unsqueeze(-1)+1).squeeze(-1)

    # position within bin
    s = (x_ins - cw) / (w + 1e-12)
    numerator = h*(s**2) + d_left*s*(1-s)
    denominator = h + (d_left + d_right - 2*h)*s*(1-s)
    y_ins = ch + (h*s + numerator/denominator) * 0  # placeholder to keep shape

    # full rational-quadratic formula
    # See NSFs paper eq. (14)
    t = s
    a = h
    b = d_left
    c = d_right
    # y within [0,1]
    y_unit = ch + a*t + (t*(1-t))*(a*(1 - a)*(2*t - 1) + (b*(1 - t) + c*t) - a)
    # derivative dy/dx
    # For stability we use automatic differentiation via torch.logsumexp-like trick is complex; use analytic from paper:
    # dy/dt = a + (1-2t)*(a*(1-a)*(2*t-1) + (b*(1-t)+c*t) - a) + t*(1-t)*(2*a*(1-a) + (c-b))
    dy_dt = a + (1-2*t)*(a*(1-a)*(2*t-1) + (b*(1-t)+c*t) - a) + t*(1-t)*(2*a*(1-a) + (c-b))
    dy_dx = dy_dt / (w + 1e-12) / (2*B)

    # assemble outputs
    y_scaled = x_scaled.clone()
    y_scaled[inside] = y_unit
    y = y_scaled*(2*B) - B

    logdet = torch.zeros_like(x)
    logdet[inside] = torch.log(torch.abs(dy_dx) + 1e-12)

    # linear tails slope=1 => logdet 0; output already set
    return y, logdet

# ---- Coupling transform ----

class RQSCoupling(nn.Module):
    def __init__(self, dim, hidden=128, K=8, B=3.0):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.net = nn.Sequential(
            nn.Linear(dim//2, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, (dim - dim//2) * (2*K + (K+1)))
        )
        self.mask_flip = False
    def set_mask_flip(self, flip: bool):
        self.mask_flip = flip
    def forward(self, z):
        if self.mask_flip:
            z1, z2 = z[:, z.size(1)//2:], z[:, :z.size(1)//2]
        else:
            z1, z2 = z[:, :z.size(1)//2], z[:, z.size(1)//2:]
        params = self.net(z1)
        D2 = z2.size(1)
        K = self.K
        widths, heights, derivatives = torch.split(params, [D2*K, D2*K, D2*(K+1)], dim=-1)
        widths = widths.view(-1, D2, K).softplus() + 1e-3
        heights = heights.view(-1, D2, K).softplus() + 1e-3
        derivatives = derivatives.view(-1, D2, K+1).softplus() + 1e-3
        y2, logdet = rational_quadratic_spline(z2, widths, heights, derivatives, B=self.B)
        if self.mask_flip:
            y = torch.cat([y2, z1], dim=1)
        else:
            y = torch.cat([z1, y2], dim=1)
        return y, logdet.sum(-1)

# ---- VAE with coupling RQS posterior ----

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

class VAE_NSF_Coupling(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, n_flows=4, K=8, B=3.0, recon='bce'):
        super().__init__()
        assert z_dim % 2 == 0, 'z_dim should be even for coupling splits'
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
        self.recon = recon
        self.flows = nn.ModuleList([RQSCoupling(z_dim, hidden=128, K=K, B=B) for _ in range(n_flows)])
        # alternate masks
        for i, f in enumerate(self.flows):
            f.set_mask_flip(bool(i % 2))
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self, x):
        mu, lv = self.enc(x)
        z = self.reparam(mu, lv)
        log_q0 = -0.5*((z-mu).pow(2)/lv.exp() + lv + torch.log(torch.tensor(2*3.141592653589793, device=x.device))).sum(-1)
        sum_logdet = torch.zeros(x.size(0), device=x.device)
        z_k = z
        for f in self.flows:
            z_k, ld = f(z_k)
            sum_logdet += ld
        x_logits = self.dec(z_k)
        log_pz = -0.5*(z_k.pow(2) + torch.log(torch.tensor(2*3.141592653589793, device=x.device))).sum(-1)
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='none').flatten(1).sum(-1)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='none').flatten(1).sum(-1)
        elbo = -rec + log_pz - (log_q0 - sum_logdet)
        return (-elbo).mean()
