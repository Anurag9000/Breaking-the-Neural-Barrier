import torch
import torch.nn.functional as F
import torch.optim as optim

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_ctrl_depth_unet import CtrlDepthUNet
from runs._common_real_image import make_real_image_loaders


def to_depth(rgb):
    return rgb.mean(dim=1, keepdim=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _, _ = make_real_image_loaders(batch_size=64, image_size=64)

    model = CtrlDepthUNet().to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        for rgb, _ in loader:
            rgb = rgb.to(device)
            dep = to_depth(rgb)
            t = torch.randint(0, cfg.T, (rgb.size(0),), device=device)
            loss = lossf(model, rgb, dep, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()

