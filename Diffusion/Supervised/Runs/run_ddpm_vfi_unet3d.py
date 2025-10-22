import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_vfi_unet3d import VFIUNet


class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=1024):
        self.prev = torch.randn(n, 3, 64, 64)
        self.mid  = torch.randn(n, 3, 64, 64)
        self.next = torch.randn(n, 3, 64, 64)

    def __len__(self):
        return len(self.prev)

    def __getitem__(self, i):
        return self.prev[i], self.mid[i], self.next[i]


loader = DataLoader(Dummy(), batch_size=32, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VFIUNet().to(device)
cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)


for epoch in range(3):
    for p, m, n in loader:
        p, m, n = p.to(device), m.to(device), n.to(device)
        t = torch.randint(0, cfg.T, (p.size(0),), device=device)
        loss = lossf(model, m, (p, n), t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}: loss {loss.item():.4f}")
