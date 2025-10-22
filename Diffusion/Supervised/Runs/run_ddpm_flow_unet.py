import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_flow_unet import FlowUNet


# -----------------------------
# Dummy dataset for optical flow
# -----------------------------
class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=1024):
        # Concatenated image pair (I1, I2)
        self.pair = torch.randn(n, 6, 128, 128)
        # Corresponding flow (u,v)
        self.flow = torch.randn(n, 2, 128, 128)

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, i):
        return self.pair[i], self.flow[i]


# DataLoader
loader = DataLoader(Dummy(), batch_size=16, shuffle=True)

# -----------------------------
# Device and model setup
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FlowUNet().to(device)

cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(3):
    for pair, flow in loader:
        pair, flow = pair.to(device), flow.to(device)
        # Random timesteps for each sample
        t = torch.randint(0, cfg.T, (pair.size(0),), device=device)
        # Forward + loss
        loss = lossf(model, flow, pair, t)
        # Backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}: loss {loss.item():.4f}")
