import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.unet1d_diff import UNet1dDiff


class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=4096, T=256):
        self.s = torch.randn(n, 1, T)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


loader = DataLoader(Dummy(), batch_size=128, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet1dDiff().to(device)

cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(3):
    for s in loader:
        s = s.to(device)  # shape: (batch, 1, T)
        t = torch.randint(0, cfg.T, (s.size(0),), device=device)
        loss = lossf(model, s, None, t)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"epoch {epoch}: {loss.item():.4f}")
