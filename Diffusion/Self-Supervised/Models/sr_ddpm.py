# models/sr_ddpm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.h_cond_unet import CondUNet
from models.h_schedules import make_beta_schedule, ddim_step_eps


class SRDDPM(nn.Module):
    """
    Super-resolution DDPM.
    Conditioned on a low-resolution input image to generate a high-resolution output.
    """

    def __init__(self, T=1000, schedule='cosine', base=64, channels=3, scale=4):
        super().__init__()
        self.T = T
        self.scale = scale

        # conditional U-Net: input = x_t + upsampled LR image
        self.net = CondUNet(base=base, in_ch=channels, cond_ch=channels, out_ch=channels)

        # diffusion schedule
        betas, alphas, alpha_bars = make_beta_schedule(T, schedule)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)

    def make_lr(self, x):
        """
        Downsample x to low resolution and then upsample back.
        This is used as the conditional input.
        """
        s = self.scale
        x_lr = F.interpolate(x, scale_factor=1.0 / s, mode='area')
        x_lr_up = F.interpolate(x_lr, scale_factor=s, mode='nearest')
        return x_lr_up

    def q_sample(self, x0, t, eps):
        """
        Forward diffusion: produce x_t given x0 and noise eps at timestep t.
        """
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def loss(self, x0):
        """
        Compute DDPM loss for super-resolution.
        """
        B = x0.size(0)
        device = x0.device

        # sample timesteps
        t = torch.randint(1, self.T, (B,), device=device)

        # noise
        eps = torch.randn_like(x0)

        # forward diffusion
        x_t = self.q_sample(x0, t, eps)

        # low-res conditional
        lr = self.make_lr(x0)

        # network input: [x_t, upsampled LR image]
        inp = torch.cat([x_t, lr], dim=1)

        # predict epsilon
        eps_hat = self.net(inp, t.float() / self.T)

        return (eps_hat - eps).pow(2).mean()

    @torch.no_grad()
    def sample(self, lr_img, steps=50):
        """
        Generate high-resolution image from low-resolution input.
        
        Args:
            lr_img: upsampled low-resolution image (B, C, H, W)
            steps: number of DDIM steps

        Returns:
            x: generated high-resolution image in [-1, 1]
        """
        B = lr_img.size(0)
        device = lr_img.device

        # initialize x_T ~ N(0, I)
        x = torch.randn_like(lr_img)

        for i in reversed(range(1, steps + 1)):
            t = torch.full((B,), int(self.T * i / steps), device=device, dtype=torch.long)
            t_prev = torch.full((B,), int(self.T * (i - 1) / steps), device=device, dtype=torch.long)

            # input for network: [x_t, upsampled LR image]
            inp = torch.cat([x, lr_img], dim=1)

            # predict noise
            eps_hat = self.net(inp, t.float() / self.T)

            # DDIM step
            x = ddim_step_eps(x, t, t_prev, eps_hat, self.alpha_bars)

        return x.clamp(-1, 1)
