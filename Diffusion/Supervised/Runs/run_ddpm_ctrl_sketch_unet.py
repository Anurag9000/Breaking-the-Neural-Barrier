import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
import torch.optim as optim
from torch.utils.data import DataLoader
from core.diffusion_core import DiffusionConfig, BetaSchedule, DiffusionLoss
from models.ddpm_ctrl_sketch_unet import CtrlSketchUNet


# ====== Dummy Sketch Dataset ======
class DummySketch(torch.utils.data.Dataset):
    def __init__(self, n=2048):
        self.rgb = torch.randn(n, 3, 64, 64)
        self.sk = (torch.randn(n, 1, 64, 64) > 0).float()

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        return self.rgb[i], self.sk[i]


# ====== Setup ======
loader = DataLoader(DummySketch(), batch_size=64, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CtrlSketchUNet().to(device)
cfg = DiffusionConfig(T=1000, objective='eps')
sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
lossf = DiffusionLoss(cfg, sched)
opt = optim.AdamW(model.parameters(), lr=2e-4)


# ====== Training Loop ======

# Init Logger

logger = ContinuousLogger(Path('results_run_ddpm_ctrl_sketch_unet'), 'run_ddpm_ctrl_sketch_unet', 'train')

for epoch in range(3):
    for rgb, sk in loader:
        rgb, sk = rgb.to(device), sk.to(device)
        t = torch.randint(0, cfg.T, (rgb.size(0),), device=device)

        loss = lossf(model, rgb, sk, t)
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