import torch
import torch.nn as nn
from models.i_common_vp import EpsUNet, beta_schedule


class VPJointClass(nn.Module):
    def __init__(self, T=1000, schedule='cosine', base=64, num_classes=10):
        super().__init__()
        self.T = T
        self.unet = EpsUNet(base=base, ch=3)
        # Tiny classifier head on low-res features
        self.clf = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes)
        )
        betas, alphas, a_bar = beta_schedule(T, schedule)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', a_bar)

    def q_sample(self, x0, t, eps):
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    def diffusion_loss(self, x0):
        B = x0.size(0)
        device = x0.device
        t = torch.randint(1, self.T, (B,), device=device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps)
        eps_hat = self.unet(x_t, t.float() / self.T)
        return (eps_hat - eps).pow(2).mean()

    def classifier_loss(self, x, y):
        return nn.functional.cross_entropy(self.clf(x), y)

    @torch.no_grad()
    def ddim_step(self, x, t, t_prev, eps_hat):
        ab_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        ab_s = self.alpha_bars[t_prev].view(-1, 1, 1, 1)
        x0 = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
        x_prev = ab_s.sqrt() * x0 + (1 - ab_s).sqrt() * eps_hat
        return x_prev

    @torch.no_grad()
    def sample(self, B, y, guidance=1.5, steps=50, device='cuda', size=(3, 32, 32)):
        x = torch.randn(B, *size, device=device)
        for i in reversed(range(1, steps + 1)):
            t = torch.full((B,), int(self.T * i / steps), device=device, dtype=torch.long)
            t_prev = torch.full((B,), int(self.T * (i - 1) / steps), device=device, dtype=torch.long)
            t01 = t.float() / self.T

            # Classifier guidance
            x.requires_grad_(True)
            logp = nn.functional.log_softmax(self.clf(x), dim=1)
            sel = logp.gather(1, y.view(-1, 1)).sum()
            grad = torch.autograd.grad(sel, x)[0]
            x = (x + guidance * grad).detach()

            # Diffusion step
            eps_hat = self.unet(x, t01)
            x = self.ddim_step(x, t, t_prev, eps_hat)

        return x.clamp(-1, 1)
