import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_sem2img_unet import Sem2ImgUNet


class DummySem(torch.utils.data.Dataset):
    def __init__(self, n=2048, k=19):
        self.rgb = torch.randn(n, 3, 64, 64)
        self.lbl = torch.randint(0, k, (n, 64, 64))
        self.k = k

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        y = F.one_hot(self.lbl[i], num_classes=self.k).permute(2, 0, 1).float()
        return self.rgb[i], y


k = 19
loader = DataLoader(DummySem(k=k), batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Sem2ImgUNet(num_classes=k).to(device)

cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(3):
    for rgb, sem in loader:
        rgb, sem = rgb.to(device), sem.to(device)
        t = torch.randint(0, cfg.T, (rgb.size(0),), device=device)
        loss = lossf(model, rgb, sem, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}: loss {loss.item():.4f}")
