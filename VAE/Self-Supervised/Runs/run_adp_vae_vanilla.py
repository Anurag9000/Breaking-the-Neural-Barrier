import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from adp_vae_vanilla import VAEConfig, VanillaVAE


def get_dataloaders(data_root: str, batch_size: int):
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tfm = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    tfm_test = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
    test = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm_test)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True), \
           DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def to_img_range(x):
    # inverse normalize approximately for visualization: assume mean/std as above
    mean = torch.tensor([[0.4914, 0.4822, 0.4465]], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([[0.2470, 0.2435, 0.2616]], device=x.device).view(1, 3, 1, 1)
    x = x * std + mean
    return torch.clamp(x, 0, 1)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    train_loader, test_loader = get_dataloaders(args.data, args.batch)

    cfg = VAEConfig(in_channels=3, img_size=32, latent_dim=args.latent, width=args.width, depth=args.depth)
    model = VanillaVAE(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        for x, _ in train_loader:
            x = x.to(device)
            # scale to [0,1] for BCE target: reverse normalize quickly
            x_bce = torch.clamp((x * torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1,3,1,1) + \
                                torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)), 0, 1)
            x_recon, mu, logvar = model(x_bce)
            losses = model.loss_fn(x_bce, x_recon, mu, logvar)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            for k in running:
                running[k] += losses[k].item()
        n = len(train_loader)
        print(f"[Epoch {epoch}] loss={running['loss']/n:.4f} recon={running['recon']/n:.4f} kl={running['kl']/n:.4f}")

        # simple eval: reconstruct a batch
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(test_loader))
            x = x.to(device)
            x_bce = torch.clamp((x * torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1,3,1,1) + \
                                torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)), 0, 1)
            x_recon, mu, logvar = model(x_bce)
            grid = torchvision.utils.make_grid(torch.cat([x_bce[:8], x_recon[:8]], dim=0), nrow=8)
            torchvision.utils.save_image(grid, out_dir / f"recon_epoch{epoch:03d}.png")

        torch.save({
            'epoch': epoch,
            'cfg': asdict(cfg),
            'state_dict': model.state_dict(),
            'opt_state': opt.state_dict()
        }, out_dir / 'checkpoint.pt')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--out', type=str, default='./runs/vae_vanilla')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--latent', type=int, default=64)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    train(args)
