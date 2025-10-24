# models/inpaint_ddpm.py
import torch
import torch.nn as nn
from models.h_cond_unet import CondUNet
from models.h_schedules import make_beta_schedule, ddim_step_eps


class InpaintDDPM(nn.Module):
    """
    Conditional DDPM for image inpainting.
    The model predicts noise only on unknown (masked) regions.
    """

    def __init__(self, T=1000, schedule='cosine', base=64, channels=3):
        super().__init__()
        self.T = T
        self.net = CondUNet(base=base, in_ch=channels, cond_ch=channels + 1, out_ch=channels)

        # diffusion schedule
        betas, alphas, alpha_bars = make_beta_schedule(T, schedule)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)

    def q_sample(self, x0, t, eps):
        """
        Forward diffusion: produce x_t given x0 and noise eps at timestep t.
        """
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return (ab.sqrt() * x0 + (1 - ab).sqrt() * eps)

    def loss(self, x0, mask):
        """
        Compute DDPM loss, only on unknown (unmasked) regions.
        """
        B = x0.size(0)
        device = x0.device

        # sample timesteps for each batch element
        t = torch.randint(1, self.T, (B,), device=device)

        # sample random noise
        eps = torch.randn_like(x0)

        # forward diffusion
        x_t = self.q_sample(x0, t, eps)

        # known context
        x_ctx = x0 * mask

        # prepare network input: [x_t, mask, context]
        inp = torch.cat([x_t, mask, x_ctx], dim=1)

        # predict epsilon
        eps_hat = self.net(inp, t.float() / self.T)

        # only penalize unknown regions
        w = (1 - mask).detach()
        return ((eps_hat - eps) * w).pow(2).mean()

    @torch.no_grad()
    def sample(self, x_init, mask, steps=50):
        """
        Perform inpainting sampling using DDIM steps.
        
        Args:
            x_init: observed context image (masked image)
            mask: 1 for known pixels, 0 for unknown
            steps: number of DDIM steps

        Returns:
            x: inpainted image in [-1,1]
        """
        B = x_init.size(0)
        device = x_init.device

        # initialize x_T ~ N(0, I)
        x = torch.randn_like(x_init)

        for i in reversed(range(1, steps + 1)):
            t = torch.full((B,), int(self.T * i / steps), device=device, dtype=torch.long)
            t_prev = torch.full((B,), int(self.T * (i - 1) / steps), device=device, dtype=torch.long)

            # known context
            x_ctx = x_init * mask

            # prepare network input
            inp = torch.cat([x, mask, x_ctx], dim=1)

            # predict noise
            eps_hat = self.net(inp, t.float() / self.T)

            # DDIM step
            x = ddim_step_eps(x, t, t_prev, eps_hat, self.alpha_bars)

            # enforce context
            x = x * (1 - mask) + x_ctx

        return x.clamp(-1, 1)
