# runs/run_scoresde.py
import os
import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
from torchvision.utils import make_grid
from models.score_sde_unified import ScoreSDEUnified
from runs._common_train_c import make_cifar10_loaders, EarlyStopper, save_samples


# -------------------------------------------------------
# Argument parsing
# -------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--type', type=str, default='ve', choices=['ve', 'vp', 'subvp'])
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--patience', type=int, default=20)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--device', type=str, default='cuda')
p.add_argument('--save', type=str, default='./results/score_sde')
p.add_argument('--steps', type=int, default=800)
args = p.parse_args([]) if __name__ == '__main__' else p.parse_args()


# -------------------------------------------------------
# Data loaders
# -------------------------------------------------------
train_loader, val_loader, _ = make_cifar10_loaders(batch_size=args.batch_size)


# -------------------------------------------------------
# Model setup
# -------------------------------------------------------
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = ScoreSDEUnified(sde_type=args.type).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
stop = EarlyStopper(patience=args.patience)


# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
best = None

# Init Logger

logger = ContinuousLogger(Path('results_run_scoresde'), 'run_scoresde', 'train')

for epoch in range(1, args.epochs + 1):
    model.train()
    for x, _ in train_loader:
        x = x.to(device)
        loss = model.loss(x)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Validation
    model.eval()
    v = 0.0
    n = 0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            v += model.loss(x).item() * x.size(0)
            n += x.size(0)
    v /= n

    # Log


    msg = f'Epoch {epoch}: val_loss={v:.4f}'


    logger.log_console(msg)


    logger.log_epoch_stats({


        "epoch": epoch,


        "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),


        "train_loss": loss.item() if 'loss' in locals() else 0


    })

    # Early stopping
    if stop.step(v):
        best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if stop.should_stop():
        break


# -------------------------------------------------------
# Load best model
# -------------------------------------------------------
if best is not None:
    model.load_state_dict(best)


# -------------------------------------------------------
# Sampling and saving
# -------------------------------------------------------
os.makedirs(args.save, exist_ok=True)

with torch.no_grad():
    # Reverse SDE sampling
    samples_sde = model.sample_sde(B=64, steps=args.steps, device=device)
    grid1 = make_grid((samples_sde + 1) / 2, nrow=8)
    save_samples(grid1, os.path.join(args.save, f'samples_scoresde_{args.type}_sde.png'))

    # ODE (probability flow) sampling
    samples_ode = model.sample_ode(B=64, steps=args.steps, device=device)
    grid2 = make_grid((samples_ode + 1) / 2, nrow=8)
    save_samples(grid2, os.path.join(args.save, f'samples_scoresde_{args.type}_ode.png'))

# Save model checkpoint
torch.save(model.state_dict(), os.path.join(args.save, f'score_sde_{args.type}.pth'))
