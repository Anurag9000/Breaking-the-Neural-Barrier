import torch
import torch.nn.functional as F
import torch.optim as optim

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_classify_map_unet import ClassifyMapUNet
from runs._common_real_image import infer_num_classes, make_real_image_loaders


def class_map_from_labels(y, h, w, k):
    return F.one_hot(y, num_classes=k).view(y.size(0), k, 1, 1).expand(y.size(0), k, h, w).float()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _, _ = make_real_image_loaders(batch_size=32, image_size=64)
    k = infer_num_classes(loader)

    model = ClassifyMapUNet(num_classes=k).to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        for _, y in loader:
            y = y.to(device)
            y_map = class_map_from_labels(y, 64, 64, k).to(device)
            t = torch.randint(0, cfg.T, (y.size(0),), device=device)
            loss = lossf(model, y_map, None, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
