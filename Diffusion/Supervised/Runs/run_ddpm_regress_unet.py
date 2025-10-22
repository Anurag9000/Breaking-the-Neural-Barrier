import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_regress_unet import RegressUNet


# -----------------------------
# Dummy Dataset (Regression Example)
# -----------------------------
class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=2048):
        # Conditioning image (e.g., RGB)
        self.img = torch.randn(n, 3, 64, 64)
        # Target to regress (e.g., depth/angle map)
        self.y = torch.randn(n, 1, 64, 64)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        return self.img[i], self.y[i]


# -----------------------------
# Setup
# -----------------------------
loader = DataLoader(Dummy(), batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RegressUNet(target_ch=1).to(device)

cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)

opt = optim.AdamW(model.parameters(), lr=2e-4)


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(3):
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Random diffusion timesteps
        t = torch.randint(0, cfg.T, (x.size(0),), device=device)

        # Compute loss (model predicts noise or v given y and x)
        loss = lossf(model, y, x, t)

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
