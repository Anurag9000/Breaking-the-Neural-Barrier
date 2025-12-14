import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_classify_map_unet import ClassifyMapUNet


# -----------------------------
# Dummy Dataset (Pixelwise Classification Example)
# -----------------------------
class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=1024, k=4):
        """
        Args:
            n: Number of samples.
            k: Number of classes (for segmentation map).
        """
        self.x = torch.randn(n, 3, 64, 64)  # input RGB images
        self.k = k
        self.y = torch.randint(0, k, (n, 64, 64))  # per-pixel class labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        # Convert integer labels → one-hot class maps
        y = torch.nn.functional.one_hot(self.y[i], num_classes=self.k)
        y = y.permute(2, 0, 1).float()  # (k, H, W)
        return self.x[i], y


# -----------------------------
# Setup
# -----------------------------
loader = DataLoader(Dummy(), batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ClassifyMapUNet(num_classes=4).to(device)

cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)

opt = optim.AdamW(model.parameters(), lr=2e-4)


# -----------------------------
# Training Loop
# -----------------------------

# Init Logger

logger = ContinuousLogger(Path('results_run_ddpm_classify_map_unet'), 'run_ddpm_classify_map_unet', 'train')

for epoch in range(3):
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Random diffusion timesteps
        t = torch.randint(0, cfg.T, (x.size(0),), device=device)

        # Diffusion loss — model predicts denoised logits given image context
        loss = lossf(model, y, None, t)

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