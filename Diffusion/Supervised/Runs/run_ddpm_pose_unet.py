import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
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

# Init Logger

logger = ContinuousLogger(Path('results_run_ddpm_pose_unet'), 'run_ddpm_pose_unet', 'train')

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

    # Log


msg = f"Epoch {epoch}: loss = {loss.item():.4f}"


    logger.log_console(msg)


    logger.log_epoch_stats({


        "epoch": epoch,


        "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),


        "train_loss": loss.item() if 'loss' in locals() else 0


    })