import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
import torch.optim as optim
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn.functional as F
from torch.utils.data import DataLoader

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.fmd_unet import SmallBackbone, FMDiffusion


# -----------------------------------------------------
# Dummy dataset (random images)
# -----------------------------------------------------
class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=2048):
        self.x = torch.randn(n, 3, 64, 64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


# -----------------------------------------------------
# Data loader
# -----------------------------------------------------
loader = DataLoader(Dummy(), batch_size=64, shuffle=True)

# -----------------------------------------------------
# Model setup
# -----------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Small backbone (feature extractor) + feature-map diffusion network
backbone = SmallBackbone().to(device)
net = FMDiffusion(feat_ch=64).to(device)

# Diffusion configuration and loss
cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)

# Optimizer
opt = optim.AdamW(list(backbone.parameters()) + list(net.parameters()), lr=2e-4)


# -----------------------------------------------------
# Training loop
# -----------------------------------------------------

# Init Logger

logger = ContinuousLogger(Path('results_run_fmd_unet'), 'run_fmd_unet', 'train')

for epoch in range(3):
    for x in loader:
        x = x.to(device)

        # Compute target clean features (no teacher; same backbone)
        with torch.no_grad():
            f_clean = backbone(x)

        # Sample random diffusion timestep
        t = torch.randint(0, cfg.T, (x.size(0),), device=device)

        # Compute diffusion loss on features
        loss = lossf(net, f_clean, None, t)

        # Backpropagation and optimization
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