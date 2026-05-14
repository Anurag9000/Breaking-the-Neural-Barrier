import torch
import torch.optim as optim

from core.sde_core import SDEConfig, ScoreLoss, probability_flow_ode, rand_times
from models.score_unet import ScoreUNet
from runs._common_real_image import make_real_image_loaders


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, _, _ = make_real_image_loaders(batch_size=128, image_size=32)

    cfg = SDEConfig(type="vp", beta_min=0.1, beta_max=20.0)
    net = ScoreUNet(img_ch=3, base=64, tdim=256).to(device)
    lossf = ScoreLoss(cfg)
    opt = optim.AdamW(net.parameters(), lr=2e-4)

    for epoch in range(5):
        for x, _ in loader:
            x = x.to(device)
            t = rand_times(x.size(0), cfg.T, device)
            loss = lossf(net, x, None, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}: loss {loss.item():.4f}")

    with torch.no_grad():
        xT = torch.randn(8, 3, 32, 32, device=device)
        x0 = probability_flow_ode(net, xT, None, cfg, steps=100)
        print("Deterministic sample shape:", x0.shape)


if __name__ == "__main__":
    main()

