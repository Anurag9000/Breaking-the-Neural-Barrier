import os
import argparse
import torch
from torchvision.utils import make_grid
from models.i_base_vp import VPBase
from models.i_energy_guided import EnergyGuided
from models.i_energies import MSEEnergy, TotalVariation
from runs._common_train_i import make_cifar10_loaders, save_samples


# -----------------------------
# Argument parsing
# -----------------------------
p = argparse.ArgumentParser()
p.add_argument('--device', type=str, default='cuda')
p.add_argument('--steps', type=int, default=50)
p.add_argument('--eta', type=float, default=0.2)
p.add_argument('--save', type=str, default='./results/partI_egd')
args = p.parse_args([]) if __name__ == '__main__' else p.parse_args()


# -----------------------------
# Data
# -----------------------------
_, _, test_loader = make_cifar10_loaders()


# -----------------------------
# Model
# -----------------------------
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = VPBase().to(device)

# Load pretrained weights if available
pth = os.path.join('./results/partI_base', 'vp_base.pth')
if os.path.exists(pth):
    model.load_state_dict(torch.load(pth, map_location=device))


# -----------------------------
# Reference target for energy
# -----------------------------
x_ref, _ = next(iter(test_loader))
x_ref = x_ref.to(device)[:64]
blur = torch.nn.AvgPool2d(3, stride=1, padding=1)
x_ref_blur = blur(x_ref)


# Energy functions
mseE = MSEEnergy()
tvE = TotalVariation(0.05)

def energy(x):
    return mseE(x, x_ref_blur) + tvE(x)


# -----------------------------
# Energy-guided sampling
# -----------------------------
guider = EnergyGuided(model, energy_fn=lambda x: energy(x), eta=args.eta)

os.makedirs(args.save, exist_ok=True)
with torch.no_grad():
    imgs = guider.sample(B=64, steps=args.steps, device=device)
    grid = make_grid((imgs + 1) / 2, nrow=8)
    save_samples(grid, os.path.join(args.save, 'samples_egd.png'))
