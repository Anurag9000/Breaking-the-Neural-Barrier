import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from core.sde_core import SDEConfig, ScoreLoss, rand_times
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
cfg = SDEConfig(type='ve', sigma_min=0.01, sigma_max=5.0)

net = ScoreUNet(img_ch=3, base=64, tdim=256, cond_dim=0).to(device)
lossf = ScoreLoss(cfg)
opt = optim.AdamW(net.parameters(), lr=2e-4)


# ====== Training Loop ======
for epoch in range(5):
    for x in loader:
        x = x.to(device)
        t = rand_times(x.size(0), cfg.T, device)

        loss = lossf(net, x, None, t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch}: loss {loss.item():.4f}")
