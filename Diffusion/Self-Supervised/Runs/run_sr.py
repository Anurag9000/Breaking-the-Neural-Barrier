import os
import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
from torchvision.utils import make_grid
from models.sr_ddpm import SRDDPM
from runs._common_train_h import make_cifar10_loaders, EarlyStopper, save_samples, make_lr

# ---------------------------
# Arguments
# ---------------------------
p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--patience', type=int, default=20)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--device', type=str, default='cuda')
p.add_argument('--save', type=str, default='./results/sr')
p.add_argument('--steps', type=int, default=50)
args = p.parse_args([]) if __name__ == '__main__' else p.parse_args()

# ---------------------------
# Data Loaders
# ---------------------------
train_loader, val_loader, _ = make_cifar10_loaders(batch_size=args.batch_size)

# ---------------------------
# Device & Model
# ---------------------------
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = SRDDPM().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
stop = EarlyStopper(patience=args.patience)

# ---------------------------
# Training Loop
# ---------------------------
best = None

# Init Logger

logger = ContinuousLogger(Path('results_run_sr'), 'run_sr', 'train')

for epoch in range(1, args.epochs + 1):
    model.train()
    for x, _ in train_loader:
        x = x.to(device)
        loss = model.loss(x)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # ---------------------------
    # Validation
    # ---------------------------
    model.eval()
    val_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            val_loss += model.loss(x).item() * x.size(0)
            n_samples += x.size(0)
    val_loss /= n_samples
    # Log

    msg = f'Epoch {epoch}: val_loss={val_loss:.4f}'

    logger.log_console(msg)

    logger.log_epoch_stats({

        "epoch": epoch,

        "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

        "train_loss": loss.item() if 'loss' in locals() else 0

    })

    # Early stopping
    if stop.step(val_loss):
        best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if stop.should_stop():
        print("Early stopping triggered.")
        break

# ---------------------------
# Load best model
# ---------------------------
if best is not None:
    model.load_state_dict(best)

# ---------------------------
# Sampling & Saving
# ---------------------------
os.makedirs(args.save, exist_ok=True)
with torch.no_grad():
    x, _ = next(iter(val_loader))
    x = x.to(device)[:64]
    lr = make_lr(x)
    samples = model.sample(lr, steps=args.steps)

    # Make grid: LR + generated SR samples
    grid = make_grid(torch.cat([(lr + 1) / 2, (samples + 1) / 2], dim=0), nrow=8)
    save_samples(grid, os.path.join(args.save, 'samples_sr.png'))

# Save model checkpoint
torch.save(model.state_dict(), os.path.join(args.save, 'sr.pth'))
