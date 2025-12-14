import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_classcond_unet import ClassCondUNet


# -----------------------------
# Dummy Dataset (for testing)
# -----------------------------
class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=1024, num_classes=10):
        self.x = torch.randn(n, 3, 32, 32)
        self.y = torch.randint(0, num_classes, (n,))
        self.num_classes = num_classes

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# -----------------------------
# Setup
# -----------------------------
num_classes = 10
loader = DataLoader(Dummy(4096, num_classes), batch_size=128, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ClassCondUNet(num_classes).to(device)

cfg = DiffusionConfig(T=1000, objective='eps', p_uncond=0.1)
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)

opt = optim.AdamW(model.parameters(), lr=1e-4)


# -----------------------------
# Training Loop
# -----------------------------

# Init Logger

logger = ContinuousLogger(Path('results_run_ddpm_classcond_unet_1'), 'run_ddpm_classcond_unet_1', 'train')

for epoch in range(5):
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # One-hot encode class labels
        y_oh = torch.nn.functional.one_hot(y, num_classes=num_classes).float()

        # Sample random diffusion timesteps
        t = torch.randint(0, cfg.T, (x.size(0),), device=device)

        # Classifier-free guidance dropout (unconditional training)
        if torch.rand(()) < cfg.p_uncond:
            y_oh = torch.zeros_like(y_oh)

        # Compute loss
        loss = lossf(model, x, y_oh, t)

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Log


msg = f"Epoch {epoch}: loss {loss.item():.4f}"


    logger.log_console(msg)


    logger.log_epoch_stats({


        "epoch": epoch,


        "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),


        "train_loss": loss.item() if 'loss' in locals() else 0


    })