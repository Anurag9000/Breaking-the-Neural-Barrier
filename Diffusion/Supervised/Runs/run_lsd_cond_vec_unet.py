import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.latent_core import DeterministicAutoencoder
from models.lsd_cond_vec_unet import LatentCondVecUNet


# -----------------------------------------------------
# Dummy dataset with class labels
# -----------------------------------------------------
class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=4096, k=10):
        self.x = torch.randn(n, 3, 64, 64)
        self.y = torch.randint(0, k, (n,))
        self.k = k

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
k = 10  # number of classes
loader = DataLoader(Dummy(k=k), batch_size=64, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Autoencoder and conditional latent diffusion U-Net
ae = DeterministicAutoencoder(z_ch=4).to(device)
net = LatentCondVecUNet(z_ch=4, cond_dim=k).to(device)

# Diffusion setup
cfg = DiffusionConfig(T=1000, objective='eps', p_uncond=0.1)
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)

# Optimizer
opt = optim.AdamW(list(ae.parameters()) + list(net.parameters()), lr=2e-4)


# -----------------------------------------------------
# Training loop
# -----------------------------------------------------

# Init Logger

logger = ContinuousLogger(Path('results_run_lsd_cond_vec_unet'), 'run_lsd_cond_vec_unet', 'train')

for epoch in range(3):
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Encode image into latent space
        z, _ = ae.encode(x)

        # Random diffusion timestep
        t = torch.randint(0, cfg.T, (x.size(0),), device=device)

        # One-hot condition vector
        c = torch.nn.functional.one_hot(y, num_classes=k).float().to(device)

        # Apply classifier-free guidance (random unconditional drop)
        if torch.rand(()) < cfg.p_uncond:
            c = torch.zeros_like(c)

        # Compute diffusion loss
        loss = lossf(net, z, c, t)

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