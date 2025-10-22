import torch, torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_med3d_unet import Med3DUNet


class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=512, vol_ch=8):
        self.vol = torch.randn(n, vol_ch, 64, 64)
        self.mask = torch.randn(n, 1, 64, 64)

    def __len__(self):
        return len(self.vol)

    def __getitem__(self, i):
        return self.vol[i], self.mask[i]


loader = DataLoader(Dummy(), batch_size=16, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Med3DUNet(vol_ch=8, out_ch=1).to(device)
cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)


for epoch in range(3):
    for vol, mask in loader:
        vol, mask = vol.to(device), mask.to(device)
        t = torch.randint(0, cfg.T, (vol.size(0),), device=device)
        loss = lossf(model, mask, vol, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}: loss {loss.item():.4f}")
