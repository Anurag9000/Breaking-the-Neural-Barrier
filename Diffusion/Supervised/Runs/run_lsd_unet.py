import torch
import torch.optim as optim

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.latent_core import DeterministicAutoencoder
from models.lsd_unet import LatentUNet
from runs._common_real_image import make_real_image_loaders


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _, _ = make_real_image_loaders(batch_size=64, image_size=64)

    ae = DeterministicAutoencoder(img_ch=3, base=64, z_ch=4).to(device)
    latent_net = LatentUNet(z_ch=4).to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(list(ae.parameters()) + list(latent_net.parameters()), lr=2e-4)

    for epoch in range(3):
        for x, _ in loader:
            x = x.to(device)
            z0, aux = ae.encode(x)
            t = torch.randint(0, cfg.T, (x.size(0),), device=device)
            loss = lossf(latent_net, z0, None, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                z, aux = ae.encode(x)
                x_hat = ae.decode(z, aux)
                recon_err = (x - x_hat).pow(2).mean().sqrt().item()
        print(f"epoch {epoch}: diff_loss={loss.item():.4f}, recon_psnr~{recon_err:.3f}")


if __name__ == "__main__":
    main()

