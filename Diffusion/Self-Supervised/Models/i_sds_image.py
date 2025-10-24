import torch


class SDSImage:
    def __init__(self, vp_model, energy_fn, w_sigma=lambda s: s**2):
        """
        Score Distillation Sampling (SDS) for images.

        vp_model: a VPBase-like diffusion model
        energy_fn: energy function to guide sampling
        w_sigma: optional scaling function for VP score term (default s^2)
        """
        self.m = vp_model
        self.energy_fn = energy_fn
        self.w_sigma = w_sigma  # Karras-style scaling; simple s^2 works

    def step(self, z, target, t01=None, lr=0.1):
        """
        Take one SDS update step on z.

        z: current image tensor
        target: target conditioning for energy function
        t01: optional time fraction tensor
        lr: learning rate for SDS step
        """
        B = z.size(0)
        device = z.device

        if t01 is None:
            t01 = torch.rand(B, device=device)

        t = (t01 * self.m.T).long().clamp(1, self.m.T - 1)

        with torch.no_grad():
            # Sample x_t from current z
            eps = torch.randn_like(z)
            x_t = self.m.q_sample(z, t, eps)  # treat z as x0

        x_t.requires_grad_(True)

        # Score gradient from diffusion model
        eps_hat = self.m.net(x_t, t01)
        s = eps_hat - eps  # proportional to score in VP parameterization

        # Energy gradient
        E = self.energy_fn(z, target)
        gE = torch.autograd.grad(E, z, retain_graph=False)[0]

        # Combined update: move z along negative of both VP score + energy gradient
        z = z - lr * (self.w_sigma(t01).view(-1, 1, 1, 1) * s.detach() + gE)

        return z.clamp(-1, 1)
