import torch
import torch.optim as optim
from torchvision import utils

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_classcond_unet import ClassCondUNet
from runs._common_real_image import infer_num_classes, make_real_image_loaders


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _, _ = make_real_image_loaders(batch_size=128, image_size=32)
    num_classes = infer_num_classes(loader)

    model = ClassCondUNet(num_classes).to(device)
    cfg = DiffusionConfig(T=1000, objective="eps", p_uncond=0.1)
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(5):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_oh = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
            t = torch.randint(0, cfg.T, (x.size(0),), device=device)
            if torch.rand(()) < cfg.p_uncond:
                y_oh = torch.zeros_like(y_oh)
            loss = lossf(model, x, y_oh, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: loss {loss.item():.4f}")

    # A minimal sample dump is enough for this runner.
    batch = next(iter(loader))
    x, y = batch[0].to(device), batch[1].to(device)
    y_oh = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    with torch.no_grad():
        t = torch.zeros(x.size(0), device=device)
        _ = model(x, t, y_oh)


if __name__ == "__main__":
    main()

