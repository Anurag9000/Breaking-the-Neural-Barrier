import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.latent_core import DeterministicAutoencoder
from models.lsd_unet import LatentUNet


# -----------------------------------------------------
# Dummy dataset (random noise for quick testing)
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

ae = DeterministicAutoencoder(img_ch=3, base=64, z_ch=4).to(device)
latent_net = LatentUNet(z_ch=4).to(device)

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
    list(ae.parameters()) + list(latent_net.parameters()),
    lr=2e-4
)

# -----------------------------------------------------
# Training loop
# -----------------------------------------------------
for epoch in range(3):
    for x in loader:
        x = x.to(device)

        # Encode image into latent space
        z0, _ = ae.encode(x)

        # Sample random diffusion timestep
        t = torch.randint(0, cfg.T, (x.size(0),), device=device)

        # Compute diffusion loss
        loss = lossf(latent_net, z0, None, t)

        # Optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Optional: stabilize AE with pixel reconstruction
        with torch.no_grad():
            z, _ = ae.encode(x)
            x_hat = ae.decode(z, _)
            recon_err = (x - x_hat).pow(2).mean().sqrt().item()

        print(f"Epoch {epoch}: diff_loss = {loss.item():.4f}, recon_psnr ~ {recon_err:.3f}")
