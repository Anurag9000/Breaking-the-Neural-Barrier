# models/self_condition_ddpm.py
import torch
import torch.nn as nn
from models.h_cond_unet import CondUNet
from models.h_schedules import make_beta_schedule, ddim_step_eps


class SelfCondDDPM(nn.Module):
    """
    Self-Conditioned DDPM: U-Net is conditioned on previous x0 predictions
    to stabilize and improve sampling.
    """

    def __init__(self, T=1000, schedule='cosine', base=64, channels=3):
        super().__init__()
        self.T = T

        # U-Net with optional self-conditioning channels (sc_ch = channels)
        self.net = CondUNet(
            base=base, in_ch=channels, cond_ch=0, sc_ch=channels, out_ch=channels
        )

        # Diffusion schedule
        betas, alphas, alpha_bars = make_beta_schedule(T, schedule)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)

    def q_sample(self, x0, t, eps):
        """
        Forward diffusion: sample x_t from x0 and noise eps at timestep t.
        """
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def loss(self, x0):
        """
        Compute self-conditioned DDPM loss.
        Half of the batch uses self-conditioning target (teacher forcing),
        the other half uses zeros for sc channels.
        """
        B, device = x0.size(0), x0.device

        # Sample random timesteps
        t = torch.randint(1, self.T, (B,), device=device)
        eps = torch.randn_like(x0)

        # Forward diffusion
        x_t = self.q_sample(x0, t, eps)

        # Decide which examples use self-conditioning
        use_sc = torch.rand(B, device=device) < 0.5
        x0_hat_prev = torch.zeros_like(x0)

        if use_sc.any():
            # Teacher forcing: estimate x0_hat from x_t (no gradient)
            with torch.no_grad():
                eps_hat0 = self.net(torch.cat([x_t, x0_hat_prev], dim=1), t.float() / self.T)
                ab_t = self.alpha_bars[t].view(-1, 1, 1, 1)
                x0_hat_prev = (x_t - (1 - ab_t).sqrt() * eps_hat0) / ab_t.sqrt()

        # Input includes self-conditioning estimate
        x_in = torch.cat([x_t, x0_hat_prev], dim=1)
        eps_hat = self.net(x_in, t.float() / self.T)

        return (eps_hat - eps).pow(2).mean()

    @torch.no_grad()
    def sample(self, B, steps=50, device='cuda', size=(3, 32, 32)):
        """
        Generate images using self-conditioning.
        """
        # Start from random noise
        x = torch.randn(B, *size, device=device)
        x0_hat = torch.zeros_like(x)

        for i in reversed(range(1, steps + 1)):
            t = torch.full((B,), int(self.T * i / steps), device=device, dtype=torch.long)
            t_prev = torch.full((B,), int(self.T * (i - 1) / steps), device=device, dtype=torch.long)

            # Predict noise
            eps_hat = self.net(torch.cat([x, x0_hat], dim=1), t.float() / self.T)

            # DDIM step
            x = ddim_step_eps(x, t, t_prev, eps_hat, self.alpha_bars)

            # Update self-conditioning estimate
            ab_t = self.alpha_bars[t].view(-1, 1, 1, 1)
            x0_hat = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()

        return x.clamp(-1, 1)
