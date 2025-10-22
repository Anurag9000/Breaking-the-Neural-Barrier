import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_audio_spec_unet import AudioSpecUNet


# -----------------------------
# Dummy Dataset for Spectrograms
# -----------------------------
class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=2048):
        # Simulate audio spectrograms: (batch, channels, freq_bins, time_steps)
        self.s = torch.randn(n, 1, 128, 256)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


# -----------------------------
# Setup
# -----------------------------
loader = DataLoader(Dummy(), batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AudioSpecUNet().to(device)

cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=1e-4)


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(3):
    for s in loader:
        s = s.to(device)

        # Random diffusion timesteps
        t = torch.randint(0, cfg.T, (s.size(0),), device=device)

        # Compute diffusion loss
        loss = lossf(model, s, None, t)

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
