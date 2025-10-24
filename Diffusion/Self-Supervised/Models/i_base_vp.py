import torch
import torch.nn as nn
from models.i_common_vp import beta_schedule, EpsUNet  # Ensure EpsUNet is implemented in i_common_vp.py

class VPBase(nn.Module):
    def __init__(self, T=1000, schedule='cosine', base=64, channels=3):
        super().__init__()
        self.T = T
        self.net = EpsUNet(base=base, ch=channels)
        betas, alphas, a_bar = beta_schedule(T, schedule)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', a_bar)

    def q_sample(self, x0, t, eps):
        """
        Diffusion forward process: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
        """
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def loss(self, x0):
        """
        Standard DDPM training loss
        """
        B = x0.size(0)
        device = x0.device
        t = torch.randint(1, self.T, (B,), device=device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps)
        eps_hat = self.net(x_t, t.float() / self.T)
        return (eps_hat - eps).pow(2).mean()

    @torch.no_grad()
    def ddim_step(self, x, t, t_prev, eps_hat):
        """
        Single DDIM reverse step
        """
        ab_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        ab_s = self.alpha_bars[t_prev].view(-1, 1, 1, 1)
        x0 = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
        x_prev = ab_s.sqrt() * x0 + (1 - ab_s).sqrt() * eps_hat
        return x_prev
