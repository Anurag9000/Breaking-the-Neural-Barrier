import os
import argparse
import torch
from torchvision.utils import save_image
from models.i_base_vp import VPBase
from models.i_sds_image import SDSImage
from models.i_energies import MSEEnergy
from runs._common_train_i import make_cifar10_loaders


# -----------------------------
# Argument parsing
# -----------------------------
p = argparse.ArgumentParser()
p.add_argument('--iters', type=int, default=500)
p.add_argument('--lr', type=float, default=0.1)
p.add_argument('--device', type=str, default='cuda')
p.add_argument('--save', type=str, default='./results/partI_sds')
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

# load trained model if present
pth = os.path.join('./results/partI_base', 'vp_base.pth')
if os.path.exists(pth):
    model.load_state_dict(torch.load(pth, map_location=device))


# -----------------------------
# Target image
# -----------------------------
x_tgt, _ = next(iter(test_loader))
x_tgt = x_tgt.to(device)[:1]


# -----------------------------
# SDS optimization
# -----------------------------
sds = SDSImage(model, energy_fn=MSEEnergy())
z = torch.randn_like(x_tgt).clamp(-1, 1).detach()


os.makedirs(args.save, exist_ok=True)
for i in range(1, args.iters + 1):
    z = sds.step(z, x_tgt, lr=args.lr)
    if i % 50 == 0:
        save_image((z.clamp(-1, 1) + 1) / 2, os.path.join(args.save, f'z_{i:04d}.png'))


# Save target and final result
save_image((x_tgt + 1) / 2, os.path.join(args.save, 'target.png'))
save_image((z.clamp(-1, 1) + 1) / 2, os.path.join(args.save, 'final.png'))
