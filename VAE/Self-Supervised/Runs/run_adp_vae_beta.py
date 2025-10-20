import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from adp_vae_beta import BetaVAEConfig, BetaVAE


MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def loaders(root, batch):
    tfm = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    tfm_t = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    tr = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    te = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tfm_t)
    return DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True), \
           DataLoader(te, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    tr, te = loaders(args.data, args.batch)

    cfg = BetaVAEConfig(latent_dim=args.latent, beta=args.beta, width=args.width, depth=args.depth)
    model = BetaVAE(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train(); sums = {"loss":0.0, "recon":0.0, "kl":0.0}
        for x,_ in tr:
            x = x.to(device)
            x_bce = torch.clamp(x * torch.tensor(STD,device=device).view(1,3,1,1) +
                                 torch.tensor(MEAN,device=device).view(1,3,1,1), 0, 1)
            xr, mu, lv = model(x_bce)
            L = model.loss_fn(x_bce, xr, mu, lv)
            opt.zero_grad(set_to_none=True); L['loss'].backward(); opt.step()
            for k in sums: sums[k]+=L[k].item()
        n=len(tr)
        print(f"[Epoch {ep}] loss={sums['loss']/n:.4f} recon={sums['recon']/n:.4f} kl={sums['kl']/n:.4f}")

        with torch.no_grad():
            x,_=next(iter(te)); x=x.to(device)
            x_bce = torch.clamp(x * torch.tensor(STD,device=device).view(1,3,1,1) +
                                 torch.tensor(MEAN,device=device).view(1,3,1,1), 0, 1)
            xr,_,_ = model(x_bce)
            grid=torchvision.utils.make_grid(torch.cat([x_bce[:8], xr[:8]],0), nrow=8)
            torchvision.utils.save_image(grid, out/f"recon_epoch{ep:03d}.png")


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='./data')
    ap.add_argument('--out', type=str, default='./runs/vae_beta')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--latent', type=int, default=32)
    ap.add_argument('--beta', type=float, default=4.0)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--cpu', action='store_true')
    args=ap.parse_args(); train(args)
