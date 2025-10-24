# models/colorize_ddpm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.h_cond_unet import CondUNet
from models.h_schedules import make_beta_schedule, ddim_step_eps


class ColorizeDDPM(nn.Module):
    """
    DDPM for image colorization.
    Conditioned on grayscale images to generate full RGB images.
    """

    def __init__(self, T=1000, schedule='cosine', base=64):
        super().__init__()
        self.T = T

        # Conditional U-Net: input = x_t + grayscale image
        self.net = CondUNet(base=base, in_ch=3, cond_ch=1, out_ch=3)

        # Diffusion schedule
        betas, alphas, alpha_bars = make_beta_schedule(T, schedule)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)

    def to_gray(self, x):
        """
        Convert RGB image to grayscale using standard luminance formula.
        Args:
            x: RGB image tensor (B, 3, H, W)
        Returns:
            gray: grayscale image (B, 1, H, W)
        """
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def q_sample(self, x0, t, eps):
        """
        Forward diffusion: produce x_t given x0 and noise eps at timestep t.
        """
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def loss(self, x0):
        """
        Compute DDPM loss for colorization.
        """
        B = x0.size(0)
        device = x0.device

        # sample timesteps
        t = torch.randint(1, self.T, (B,), device=device)

        # noise
        eps = torch.randn_like(x0)

        # forward diffusion
        x_t = self.q_sample(x0, t, eps)

        # grayscale conditional
        gray = self.to_gray(x0)

        # network input: [x_t, gray]
        inp = torch.cat([x_t, gray], dim=1)

        # predict noise
        eps_hat = self.net(inp, t.float() / self.T)

        return (eps_hat - eps).pow(2).mean()

    @torch.no_grad()
    def sample(self, gray, steps=50):
        """
        Generate colorized images from grayscale input.
        Args:
            gray: grayscale image tensor (B, 1, H, W)
            steps: number of DDIM steps
        Returns:
            x: generated RGB image in [-1, 1]
        """
        B = gray.size(0)
        device = gray.device

        # initialize x_T ~ N(0, I)
        x = torch.randn(B, 3, gray.size(2), gray.size(3), device=device)

        for i in reversed(range(1, steps + 1)):
            t = torch.full((B,), int(self.T * i / steps), device=device, dtype=torch.long)
            t_prev = torch.full((B,), int(self.T * (i - 1) / steps), device=device, dtype=torch.long)

            # network input: [x_t, gray]
            inp = torch.cat([x, gray], dim=1)

            # predict noise
            eps_hat = self.net(inp, t.float() / self.T)

            # DDIM step
            x = ddim_step_eps(x, t, t_prev, eps_hat, self.alpha_bars)

        return x.clamp(-1, 1)
