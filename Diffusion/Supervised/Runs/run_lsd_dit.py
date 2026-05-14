import torch
import torch.optim as optim

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.latent_core import DeterministicAutoencoder
from models.lsd_dit import DiTLatent
from runs._common_real_image import make_real_image_loaders


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _, _ = make_real_image_loaders(batch_size=64, image_size=64)

    ae = DeterministicAutoencoder(z_ch=4).to(device)
    net = DiTLatent(z_ch=4).to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(list(ae.parameters()) + list(net.parameters()), lr=2e-4)

    for epoch in range(3):
        for x, _ in loader:
            x = x.to(device)
            z, aux = ae.encode(x)
            t = torch.randint(0, cfg.T, (x.size(0),), device=device)
            loss = lossf(net, z, None, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()

