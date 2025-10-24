# models/studentt_diffusion_model.py
import torch
import torch.nn as nn
from models.e_common_unet import UNetE


class StudentTDiffusion(nn.Module):
    def __init__(self, T=1000, base=64, channels=3, nu=3.0, s_min=1e-3, s_max=0.5):
        super().__init__()
        self.T = T
        self.nu = nu
        self.s_min = s_min
        self.s_max = s_max
        self.net = UNetE(base=base, in_ch=channels, out_ch=channels)

    def s(self, t):
        """
        Scale schedule between s_min and s_max
        """
        return self.s_min + (self.s_max - self.s_min) * (t.float() / self.T)

    def student_t_noise(self, x, scale):
        """
        Gaussian scale mixture for Student-t noise:
        z ~ N(0,1), u ~ Chi2(nu)/nu ⇒ noise = z * sqrt(scale^2 * nu / u)
        """
        z = torch.randn_like(x)
        u = torch.distributions.Chi2(df=torch.tensor(self.nu)).sample(z.shape).to(z.device)
        return z * torch.sqrt((scale**2) * (self.nu / (u + 1e-8)))

    def q_sample(self, x0, t, noise):
        """
        Variance-preserving style blend for Student-t diffusion
        """
        s = self.s(t).view(-1, 1, 1, 1)
        alpha = torch.exp(-s)
        return alpha * x0 + (1 - alpha) * noise

    def loss(self, x0):
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)
        noise = self.student_t_noise(x0, 1.0)
        x_t = self.q_sample(x0, t, noise)
        eps_hat = self.net(x_t, t)
        # Use robust Huber loss to stabilize heavy tails
        return torch.nn.functional.smooth_l1_loss(eps_hat, noise)

    @torch.no_grad()
    def sample(self, B, steps=100, device='cuda', size=(3, 32, 32)):
        x = torch.randn(B, *size, device=device)
        for s in reversed(range(steps)):
            t = torch.full((B,), int(self.T * s / steps), device=device, dtype=torch.long)
            eps = self.net(x, t)
            alpha = torch.exp(-self.s(t)).view(-1, 1, 1, 1)
            x0_hat = (x - (1 - alpha) * eps) / (alpha + 1e-8)
            x = alpha * x + (1 - alpha) * x0_hat
            # Mild corrector
            x = x + torch.randn_like(x) * 0.01
        return x.clamp(-1, 1)
