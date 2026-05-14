import torch
import torch.optim as optim
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.convnext_diff import ConvNeXtDiff
from runs._common_real_image import make_real_image_loaders

loader, _, _ = make_real_image_loaders(batch_size=64, image_size=64)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConvNeXtDiff().to(device)

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
