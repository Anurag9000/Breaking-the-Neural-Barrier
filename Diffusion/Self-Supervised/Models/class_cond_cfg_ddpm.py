# models/class_cond_cfg_ddpm.py
import torch
import torch.nn as nn
from models.h_cond_unet import CondUNet
from models.h_schedules import make_beta_schedule, ddim_step_eps


class ClassCondCFGDDPM(nn.Module):
    """
    Class-conditional DDPM with classifier-free guidance (CFG).
    Supports optional dropout of class conditioning during training.
    """

    def __init__(self, num_classes=10, T=1000, schedule='cosine', base=64):
        super().__init__()
        self.T = T
        self.num_classes = num_classes

        # Conditional U-Net with optional null embedding for CFG
        self.net = CondUNet(
            base=base, in_ch=3, cond_ch=0, out_ch=3,
            num_classes=num_classes, cfg_null=True
        )

        # Diffusion schedule
        betas, alphas, alpha_bars = make_beta_schedule(T, schedule)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)

    def q_sample(self, x0, t, eps):
        """
        Forward diffusion: produce x_t given x0 and noise eps at timestep t.
        """
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def loss(self, x0, y):
        """
        Compute DDPM loss with optional classifier-free dropout for CFG.
        Args:
            x0: original image tensor (B,3,H,W)
            y: class labels tensor (B,)
        """
        B, device = x0.size(0), x0.device

        # Sample random timesteps
        t = torch.randint(1, self.T, (B,), device=device)

        # Noise
        eps = torch.randn_like(x0)

        # Forward diffusion
        x_t = self.q_sample(x0, t, eps)

        # Classifier-free dropout
        drop = torch.rand(B, device=device) < 0.1
        y_in = y.clone()
        y_in[drop] = self.num_classes  # null class id

        # Predict noise
        eps_hat = self.net(x_t, t.float() / self.T, y_in)

        return (eps_hat - eps).pow(2).mean()

    @torch.no_grad()
    def sample(self, B, y, guidance=1.5, steps=50, device='cuda', size=(3, 32, 32)):
        """
        Sample images conditioned on class labels using classifier-free guidance.
        Args:
            B: batch size
            y: class labels tensor (B,)
            guidance: CFG scale (>1 strengthens conditioning)
            steps: number of DDIM steps
            device: device
            size: output image shape
        Returns:
            x: generated images in [-1,1]
        """
        # Initialize x_T ~ N(0, I)
        x = torch.randn(B, *size, device=device)

        for i in reversed(range(1, steps + 1)):
            t = torch.full((B,), int(self.T * i / steps), device=device, dtype=torch.long)
            t_prev = torch.full((B,), int(self.T * (i - 1) / steps), device=device, dtype=torch.long)

            # Unconditional prediction (CFG)
            eps_u = self.net(x, t.float() / self.T, torch.full_like(y, self.num_classes))

            # Conditional prediction
            eps_c = self.net(x, t.float() / self.T, y)

            # Classifier-free guidance
            eps_hat = (1 + guidance) * eps_c - guidance * eps_u

            # DDIM step
            x = ddim_step_eps(x, t, t_prev, eps_hat, self.alpha_bars)

        return x.clamp(-1, 1)
