import torch
import torch.nn.functional as F
import torch.optim as optim

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.fmd_unet import SmallBackbone, FMDiffusion
from runs._common_real_image import make_real_image_loaders


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _, _ = make_real_image_loaders(batch_size=64, image_size=64)

    backbone = SmallBackbone().to(device)
    net = FMDiffusion(feat_ch=64).to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(list(backbone.parameters()) + list(net.parameters()), lr=2e-4)

    for epoch in range(3):
        for x, _ in loader:
            x = x.to(device)
            with torch.no_grad():
                f_clean = backbone(x)
            t = torch.randint(0, cfg.T, (x.size(0),), device=device)
            loss = lossf(net, f_clean, None, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()

