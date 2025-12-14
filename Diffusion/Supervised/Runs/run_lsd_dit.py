import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.latent_core import DeterministicAutoencoder
from models.lsd_dit import DiTLatent


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

ae = DeterministicAutoencoder(z_ch=4).to(device)
net = DiTLatent(z_ch=4).to(device)

# -----------------------------------------------------
# Diffusion configuration
# -----------------------------------------------------
cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)

# -----------------------------------------------------
# Optimizer
# -----------------------------------------------------
opt = optim.AdamW(
    list(ae.parameters()) + list(net.parameters()),
    lr=2e-4
)

# -----------------------------------------------------
# Training loop
# -----------------------------------------------------

# Init Logger

logger = ContinuousLogger(Path('results_run_lsd_dit'), 'run_lsd_dit', 'train')

for epoch in range(3):
    for x in loader:
        x = x.to(device)

        # Encode image into latent space
        z, _ = ae.encode(x)

        # Sample random diffusion timestep
        t = torch.randint(0, cfg.T, (x.size(0),), device=device)

        # Compute diffusion loss on latent representation
        loss = lossf(net, z, None, t)

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