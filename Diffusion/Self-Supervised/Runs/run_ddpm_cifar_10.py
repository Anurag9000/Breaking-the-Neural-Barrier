import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from utils.adp_logging import ContinuousLogger
from torchvision import utils as tvutils

from ddpm_model import UNet, DDPM, DiffusionConfig, count_parameters
from runs._common_real_image import make_real_image_loaders

# -----------------------------
# Repro & small utilities
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Training / evaluation (self-supervised loss on val)
# -----------------------------

def train_one_epoch(model: DDPM, loader, optimizer, device):
    model.train()
    total = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        # diffusion expects inputs in [-1,1], already normalized
        t = torch.randint(0, model.cfg.timesteps, (imgs.size(0),), device=device)
        loss = model.p_losses(imgs, t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model: DDPM, loader, device):
    model.eval()
    total = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        t = torch.randint(0, model.cfg.timesteps, (imgs.size(0),), device=device)
        loss = model.p_losses(imgs, t)
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def save_samples(model: DDPM, out_dir: Path, device, n: int = 64):
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = model.sample((n, 3, 32, 32), device=device)
    # unnormalize from [-1,1] to [0,1]
    grid = (samples + 1) / 2
    tvutils.save_image(grid, out_dir / 'samples.png', nrow=8)

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--save-dir', type=str, default='./artifacts_ddpm')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--val-split', type=int, default=5000)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--beta-start', type=float, default=1e-4)
    parser.add_argument('--beta-end', type=float, default=2e-2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    (save_dir / 'ckpts').mkdir(parents=True, exist_ok=True)
    (save_dir / 'samples').mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    train_loader, val_loader, _ = make_real_image_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_ratio=0.1,
        test_ratio=0.1,
        num_workers=0,
        image_size=32,
    )

    # Model
    unet = UNet(in_ch=3, base=64, ch_mult=(1,2,4), time_dim=256, out_ch=3)
    diff = DDPM(unet, DiffusionConfig(timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)).to(device)

    print(f"Parameters: {count_parameters(diff)/1e6:.2f} M")

    optimizer = torch.optim.AdamW(diff.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float('inf')
    best_path = save_dir / 'ckpts' / 'best.pt'
    patience = 30
    bad = 0


    # Init Logger


    logger = ContinuousLogger(Path('results_run_ddpm_cifar_10'), 'run_ddpm_cifar_10', 'train')


    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(diff, train_loader, optimizer, device)
        va_loss = evaluate(diff, val_loader, device)
        # Log

        msg = f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })

        if va_loss + 1e-12 < best_val:
            best_val = va_loss
            bad = 0
            torch.save({'model': diff.state_dict(), 'cfg': vars(args)}, best_path)
        else:
            bad += 1

        if epoch % 20 == 0:
            save_samples(diff, save_dir / 'samples', device, n=64)

        if bad >= patience:
            print("Early stopping triggered.")
            break

    # Load best and export final samples
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        diff.load_state_dict(state['model'])
    save_samples(diff, save_dir / 'samples', device, n=64)

    print("Done. Artifacts in:", str(save_dir))

if __name__ == '__main__':
    main()
