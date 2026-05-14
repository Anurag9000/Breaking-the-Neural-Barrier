import torch
import torch.optim as optim
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.unet_blocks import SimpleUNet
from runs._common_real_image import make_real_image_loaders

loader, _, _ = make_real_image_loaders(batch_size=128, image_size=32)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleUNet(in_ch=3, out_ch=3, base=64, tdim=256, cond_dim=0).to(device)

cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(3):
    for x, _ in loader:
        x = x.to(device)
        t = torch.randint(0, cfg.T, (x.size(0),), device=device)
        loss = lossf(model, x, None, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}: loss {loss.item():.4f}")
