import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_xdomain_unet import XDomainUNet


class DummyXDomain(torch.utils.data.Dataset):
    def __init__(self, n=2048):
        # Example: NIR(1ch) → RGB(3ch)
        self.x = torch.randn(n, 1, 64, 64)  # source domain
        self.y = torch.randn(n, 3, 64, 64)  # target domain

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


loader = DataLoader(DummyXDomain(), batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = XDomainUNet(y_ch=3, x_ch=1).to(device)

cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(3):
    for x_src, y_tgt in loader:
        x_src, y_tgt = x_src.to(device), y_tgt.to(device)
        t = torch.randint(0, cfg.T, (x_src.size(0),), device=device)
        loss = lossf(model, y_tgt, x_src, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}: loss {loss.item():.4f}")
