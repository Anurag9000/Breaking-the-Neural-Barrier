import torch
import torch.nn.functional as F
import torch.optim as optim

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_ctrl_sketch_unet import CtrlSketchUNet
from runs._common_real_image import make_real_image_loaders


def sketch_from_rgb(rgb):
    gray = rgb.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=rgb.device, dtype=rgb.dtype).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
    return (mag > mag.mean(dim=(2, 3), keepdim=True)).float()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _, _ = make_real_image_loaders(batch_size=64, image_size=64)

    model = CtrlSketchUNet().to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        for rgb, _ in loader:
            rgb = rgb.to(device)
            sk = sketch_from_rgb(rgb)
            t = torch.randint(0, cfg.T, (rgb.size(0),), device=device)
            loss = lossf(model, rgb, sk, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()

