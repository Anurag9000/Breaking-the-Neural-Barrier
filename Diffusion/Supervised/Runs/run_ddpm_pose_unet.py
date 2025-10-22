import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_pose_unet import PoseUNet


# -----------------------------
# Dummy Dataset for Pose Heatmaps
# -----------------------------
class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=1024, joints=17):
        """
        Args:
            n: Number of samples
            joints: Number of joints / heatmap channels
        """
        self.img = torch.randn(n, 3, 128, 128)      # Input images
        self.hm = torch.randn(n, joints, 128, 128)  # Target heatmaps

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        return self.img[i], self.hm[i]


# -----------------------------
# Setup
# -----------------------------
loader = DataLoader(Dummy(), batch_size=16, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PoseUNet(joints=17).to(device)

cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(3):
    for img, hm in loader:
        img, hm = img.to(device), hm.to(device)

        # Random diffusion timesteps
        t = torch.randint(0, cfg.T, (img.size(0),), device=device)

        # Compute diffusion loss
        loss = lossf(model, hm, img, t)  # model learns to denoise heatmaps conditioned on images

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
