import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_textcond_unet import TextCondUNet


# ====== Dummy Text-Image Dataset ======
class DummyTextImg(torch.utils.data.Dataset):
    def __init__(self, n=2048, text_dim=512):
        self.x = torch.randn(n, 3, 64, 64)
        self.tvec = torch.randn(n, text_dim)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.tvec[i]


# ====== Setup ======
text_dim = 512
loader = DataLoader(DummyTextImg(text_dim=text_dim), batch_size=64, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = TextCondUNet(text_dim=text_dim).to(device)
cfg = DiffusionConfig(T=1000, objective='eps', p_uncond=0.1)
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)


# ====== Training Loop ======

# Init Logger

logger = ContinuousLogger(Path('results_run_ddpm_textcond_unet'), 'run_ddpm_textcond_unet', 'train')

for epoch in range(3):
    for x, txt in loader:
        x, txt = x.to(device), txt.to(device)
        t = torch.randint(0, cfg.T, (x.size(0),), device=device)

        # Classifier-free guidance dropout
        if torch.rand(()) < cfg.p_uncond:
            txt = torch.zeros_like(txt)

        loss = lossf(model, x, txt, t)
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