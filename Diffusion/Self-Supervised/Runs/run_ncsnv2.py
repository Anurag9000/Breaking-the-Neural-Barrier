# runs/run_ncsnv2.py
import os
import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
from torchvision.utils import make_grid
from models.ncsnv2_model import NCSNv2
from runs._common_train_b import make_cifar10_loaders, EarlyStopper, save_samples


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--patience', type=int, default=25)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--spectral_norm', action='store_true')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--save', type=str, default='./results/ncsnv2')
parser.add_argument('--steps_per_level', type=int, default=100)
args = parser.parse_args([]) if __name__ == '__main__' else parser.parse_args()


# Data loaders
train_loader, val_loader, _ = make_cifar10_loaders(batch_size=args.batch_size)

# Device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Model and optimizer
model = NCSNv2(spectral_norm=args.spectral_norm).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
stop = EarlyStopper(patience=args.patience)

best_state = None


# Init Logger


logger = ContinuousLogger(Path('results_run_ncsnv2'), 'run_ncsnv2', 'train')


for epoch in range(1, args.epochs + 1):
    # Training
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
    vloss = 0.0
    n = 0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            vloss += model.loss(x).item() * x.size(0)
            n += x.size(0)
    vloss /= n

    # Log


    msg = f'Epoch {epoch}: val_loss={vloss:.4f}'


    logger.log_console(msg)


    logger.log_epoch_stats({


        "epoch": epoch,


        "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),


        "train_loss": loss.item() if 'loss' in locals() else 0


    })

    # Early stopping
    if stop.step(vloss):
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if stop.should_stop():
        break

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)

# Save samples and model
os.makedirs(args.save, exist_ok=True)
with torch.no_grad():
    samples = model.sample(B=64, steps_per_level=args.steps_per_level, device=device)
    grid = make_grid((samples + 1) / 2, nrow=8)
    save_samples(grid, os.path.join(args.save, 'samples_ncsnv2.png'))
    torch.save(model.state_dict(), os.path.join(args.save, 'ncsnv2.pth'))
