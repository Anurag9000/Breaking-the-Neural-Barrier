import torch
import torch.optim as optim

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.latent_core import DeterministicAutoencoder
from models.lsd_cond_vec_unet import LatentCondVecUNet
from runs._common_real_image import infer_num_classes, make_real_image_loaders


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _, _ = make_real_image_loaders(batch_size=64, image_size=64)
    num_classes = infer_num_classes(loader)

    ae = DeterministicAutoencoder(z_ch=4).to(device)
    net = LatentCondVecUNet(z_ch=4, cond_dim=num_classes).to(device)
    cfg = DiffusionConfig(T=1000, objective="eps", p_uncond=0.1)
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(list(ae.parameters()) + list(net.parameters()), lr=2e-4)

    for epoch in range(3):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z, _ = ae.encode(x)
            t = torch.randint(0, cfg.T, (x.size(0),), device=device)
            c = torch.nn.functional.one_hot(y, num_classes=num_classes).float().to(device)
            if torch.rand(()) < cfg.p_uncond:
                c = torch.zeros_like(c)
            loss = lossf(net, z, c, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()

