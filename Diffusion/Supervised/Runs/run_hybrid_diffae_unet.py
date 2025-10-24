import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.hybrid_diffae_unet import HybridDiffAE


# -----------------------------------------------------
# Dummy dataset (random image samples)
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
model = HybridDiffAE().to(device)

# -----------------------------------------------------
# Diffusion setup
# -----------------------------------------------------
cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)

# -----------------------------------------------------
# Optimizer and loss weights
# -----------------------------------------------------
opt = optim.AdamW(model.parameters(), lr=2e-4)
lambda_recon = 0.1  # balance factor between diffusion and reconstruction losses

# -----------------------------------------------------
# Training loop with average loss tracking
# -----------------------------------------------------
for epoch in range(3):
    total_loss = 0.0
    total_diff_loss = 0.0
    total_recon_loss = 0.0
    num_batches = 0

    for x in loader:
        x = x.to(device)

        # Encode image to latent space
        z, _ = model.ae.encode(x)

        # Random diffusion timestep
        t = torch.randint(0, cfg.T, (x.size(0),), device=device)

        # Diffusion loss in latent space
        diff_loss = lossf(model.denoiser, z, None, t)

        # Reconstruction loss in pixel space
        x_hat = model.ae.decode(z, _)
        recon = F.l1_loss(x_hat, x)

        # Total loss
        loss = diff_loss + lambda_recon * recon

        # Backpropagation and optimization
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Accumulate losses for averaging
        total_loss += loss.item()
        total_diff_loss += diff_loss.item()
        total_recon_loss += recon.item()
        num_batches += 1

    # Print average losses for this epoch
    print(
        f"Epoch {epoch}: avg_total_loss = {total_loss / num_batches:.4f} "
        f"(avg_diff = {total_diff_loss / num_batches:.4f}, "
        f"avg_recon = {total_recon_loss / num_batches:.4f})"
    )
