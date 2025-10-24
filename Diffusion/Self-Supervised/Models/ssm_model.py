# models/ssm_model.py
import torch
import torch.nn as nn
from models.sm_common_unet import ScoreUNet


class SSM(nn.Module):
    """
    Sliced Score Matching (SSM) model using random projections.
    """

    def __init__(self, sigma=0.2, base=64, channels=3, num_projections=64):
        super().__init__()
        self.sigma = sigma
        self.num_projections = num_projections
        self.net = ScoreUNet(base=base, in_ch=channels, out_ch=channels)

    def loss(self, x):
        """
        Sliced score matching loss:
        project score onto random directions and minimize squared error.
        """
        B, C, H, W = x.shape
        device = x.device

        sigma = torch.full((B,), self.sigma, device=device)
        noise = torch.randn_like(x) * self.sigma
        x_t = x + noise

        # true score of Gaussian corrupted x
        true_score = -noise / (self.sigma ** 2)
        pred_score = self.net(x_t, sigma)

        # random unit projections per sample
        v = torch.randn(B, self.num_projections, C, H, W, device=device)
        v_flat = v.flatten(2)
        v = v / v_flat.norm(dim=2, keepdim=True).unsqueeze(-1)

        # project both scores along v and compute L2 loss
        ps = (pred_score.unsqueeze(1) * v).flatten(2).sum(dim=2)
        ts = (true_score.unsqueeze(1) * v).flatten(2).sum(dim=2)

        return (ps - ts).pow(2).mean()

    @torch.no_grad()
    def langevin_sample(self, B, steps=200, step_size=0.01, device='cuda', size=(3, 32, 32)):
        """
        Generate samples via Langevin dynamics.
        """
        x = torch.randn(B, *size, device=device)
        sigma = torch.full((B,), self.sigma, device=device)

        for _ in range(steps):
            grad = self.net(x, sigma)
            x = x + 0.5 * step_size * grad + (step_size ** 0.5) * torch.randn_like(x)

        return x.clamp(-1, 1)
