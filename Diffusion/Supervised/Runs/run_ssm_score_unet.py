import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
import torch.optim as optim
from torch.utils.data import DataLoader
from core.sde_core import SDEConfig, ScoreLoss, rand_times
from models.score_unet import ScoreUNet


# ====== Dummy Dataset ======
class Dummy(torch.utils.data.Dataset):
    def __init__(self, n=4096, k=10):
        self.x = torch.randn(n, 3, 32, 32)
        self.y = torch.randint(0, k, (n,))
        self.k = k

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ====== Setup ======
k = 10
loader = DataLoader(Dummy(k=k), batch_size=128, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = SDEConfig(type='vp', p_uncond=0.1)
net = ScoreUNet(img_ch=3, base=64, tdim=256, cond_dim=k).to(device)
lossf = ScoreLoss(cfg)
opt = optim.AdamW(net.parameters(), lr=2e-4)


# ====== Training Loop ======

# Init Logger

logger = ContinuousLogger(Path('results_run_ssm_score_unet'), 'run_ssm_score_unet', 'train')

for epoch in range(5):
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        t = rand_times(x.size(0), cfg.T, device)
        c = torch.nn.functional.one_hot(y, num_classes=k).float().to(device)

        # Classifier-free guidance dropout
        if torch.rand(()) < cfg.p_uncond:
            c = torch.zeros_like(c)

        loss = lossf(net, x, c, t)
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