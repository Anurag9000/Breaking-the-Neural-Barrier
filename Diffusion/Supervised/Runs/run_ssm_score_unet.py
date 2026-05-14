import torch
import torch.optim as optim
import torch.nn.functional as F

from core.sde_core import SDEConfig, ScoreLoss, rand_times
from models.score_unet import ScoreUNet
from runs._common_real_image import infer_num_classes, make_real_image_loaders


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _, _ = make_real_image_loaders(batch_size=128, image_size=32)
    num_classes = infer_num_classes(loader)

    cfg = SDEConfig(type="vp", p_uncond=0.1)
    net = ScoreUNet(img_ch=3, base=64, tdim=256, cond_dim=num_classes).to(device)
    lossf = ScoreLoss(cfg)
    opt = optim.AdamW(net.parameters(), lr=2e-4)

    for epoch in range(5):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t = rand_times(x.size(0), cfg.T, device)
            c = F.one_hot(y, num_classes=num_classes).float().to(device)
            if torch.rand(()) < cfg.p_uncond:
                c = torch.zeros_like(c)
            loss = lossf(net, x, c, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()

