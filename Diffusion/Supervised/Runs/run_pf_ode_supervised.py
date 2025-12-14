import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
import torch.optim as optim
from torch.utils.data import DataLoader
from core.sde_core import (
    SDEConfig,
    ScoreLoss,
    rand_times,
    probability_flow_ode,
)
from models.score_unet import ScoreUNet


# ====== Dummy Dataset ======
class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=4096):
        self.x = torch.randn(n, 3, 32, 32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


# ====== Setup ======
loader = DataLoader(Dummy(), batch_size=128, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = SDEConfig(type='vp', beta_min=0.1, beta_max=20.0)

net = ScoreUNet(img_ch=3, base=64, tdim=256).to(device)
lossf = ScoreLoss(cfg)
opt = optim.AdamW(net.parameters(), lr=2e-4)


# ====== Training Loop ======

# Init Logger

logger = ContinuousLogger(Path('results_run_pf_ode_supervised'), 'run_pf_ode_supervised', 'train')

for epoch in range(5):
    for x in loader:
        x = x.to(device)
        t = rand_times(x.size(0), cfg.T, device)

        loss = lossf(net, x, None, t)
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


# ====== Deterministic Sampling Demo ======
with torch.no_grad():
    xT = torch.randn(8, 3, 32, 32, device=device)
    x0 = probability_flow_ode(net, xT, None, cfg, steps=100)
    print("Deterministic sample shape:", x0.shape)