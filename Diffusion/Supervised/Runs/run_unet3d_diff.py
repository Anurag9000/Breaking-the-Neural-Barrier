import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.unet3d_diff import UNet3dDiff


class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=512):
        self.vol = torch.randn(n, 1, 16, 64, 64)

    def __len__(self):
        return len(self.vol)

    def __getitem__(self, i):
        return self.vol[i]


loader = DataLoader(Dummy(), batch_size=8, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet3dDiff().to(device)

cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(3):
    for v in loader:
        v = v.to(device)
        t = torch.randint(0, cfg.T, (v.size(0),), device=device)
        loss = lossf(model, v, None, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}: {loss.item():.4f}")
