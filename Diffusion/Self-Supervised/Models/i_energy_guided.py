import torch
from models.i_base_vp import VPBase


class EnergyGuided:
    def __init__(self, model: VPBase, energy_fn, eta=0.2):
        self.m = model
        self.energy_fn = energy_fn
        self.eta = eta

    @torch.no_grad()
    def sample(self, B, steps=50, device='cuda', size=(3, 32, 32), energy_kwargs=None):
        x = torch.randn(B, *size, device=device)
        for i in reversed(range(1, steps + 1)):
            t = torch.full((B,), int(self.m.T * i / steps), device=device, dtype=torch.long)
            t_prev = torch.full((B,), int(self.m.T * (i - 1) / steps), device=device, dtype=torch.long)
            t01 = t.float() / self.m.T

            # Predict noise
            eps_hat = self.m.net(x, t01)

            # Guidance: gradient step on energy
            x.requires_grad_(True)
            E = self.energy_fn(x, **(energy_kwargs or {}))
            g = torch.autograd.grad(E, x)[0]
            x = (x - self.eta * g).detach()

            # DDIM update
            x = self.m.ddim_step(x, t, t_prev, eps_hat)

        return x.clamp(-1, 1)
