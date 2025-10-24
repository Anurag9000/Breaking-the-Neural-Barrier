import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.hybrid_ct_diff import HybridCTDiff


class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=4096):
        self.x = torch.randn(n, 3, 64, 64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


loader = DataLoader(Dummy(), batch_size=64, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HybridCTDiff().to(device)

cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(3):
    for x in loader:
        x = x.to(device)
        t = torch.randint(0, cfg.T, (x.size(0),), device=device)
        loss = lossf(model, x, None, t)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"epoch {epoch}: {loss.item():.4f}")
