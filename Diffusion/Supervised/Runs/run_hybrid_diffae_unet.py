import torch
import torch.nn.functional as F
import torch.optim as optim

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.hybrid_diffae_unet import HybridDiffAE
from runs._common_real_image import make_real_image_loaders


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _, _ = make_real_image_loaders(batch_size=64, image_size=64)

    model = HybridDiffAE().to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)
    lambda_recon = 0.1

    for epoch in range(3):
        for x, _ in loader:
            x = x.to(device)
            z, aux = model.ae.encode(x)
            t = torch.randint(0, cfg.T, (x.size(0),), device=device)
            diff_loss = lossf(model.denoiser, z, None, t)
            x_hat = model.ae.decode(z, aux)
            recon = F.l1_loss(x_hat, x)
            loss = diff_loss + lambda_recon * recon
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(
            f"epoch {epoch}: total={loss.item():.4f} diff={diff_loss.item():.4f} recon={recon.item():.4f}"
        )


if __name__ == "__main__":
    main()

