import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_ctrl_depth_unet import CtrlDepthUNet


class DummyDepth(torch.utils.data.Dataset):
    def __init__(self, n=2048):
        self.rgb = torch.randn(n, 3, 64, 64)
        self.dep = torch.rand(n, 1, 64, 64)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        return self.rgb[i], self.dep[i]


loader = DataLoader(DummyDepth(), batch_size=64, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CtrlDepthUNet().to(device)
cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(3):
    for rgb, dep in loader:
        rgb, dep = rgb.to(device), dep.to(device)
        t = torch.randint(0, cfg.T, (rgb.size(0),), device=device)
        loss = lossf(model, rgb, dep, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}: loss {loss.item():.4f}")
