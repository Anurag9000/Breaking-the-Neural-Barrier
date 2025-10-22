import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_timeseries_1d_unet import TS1DUNet


# Dummy 1D time-series dataset
class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=4096, T=256):
        self.s = torch.randn(n, 1, 1, T)  # shape: (batch, ch, 1, T)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


# DataLoader
loader = DataLoader(Dummy(), batch_size=64, shuffle=True)


# Device and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TS1DUNet().to(device)

# Diffusion config, scheduler, and loss
cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)

# Optimizer
opt = optim.AdamW(model.parameters(), lr=1e-4)


# Training loop
for epoch in range(3):
    for s in loader:
        s = s.to(device)
        t = torch.randint(0, cfg.T, (s.size(0),), device=device)
        loss = lossf(model, s, None, t)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    print(f"epoch {epoch}: loss {loss.item():.4f}")
