import os
import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
from torchvision.utils import make_grid
from models.i_joint_classifier import VPJointClass
from runs._common_train_i import make_cifar10_loaders, EarlyStopper, save_samples


# -----------------------------
# Argument parsing
# -----------------------------
p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--patience', type=int, default=20)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--device', type=str, default='cuda')
p.add_argument('--save', type=str, default='./results/partI_jcg')
p.add_argument('--steps', type=int, default=50)
p.add_argument('--guidance', type=float, default=1.5)
args = p.parse_args([]) if __name__ == '__main__' else p.parse_args()


# -----------------------------
# Data
# -----------------------------
train_loader, val_loader, _ = make_cifar10_loaders(batch_size=args.batch_size)


# -----------------------------
# Model and optimizer
# -----------------------------
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = VPJointClass().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
stop = EarlyStopper(patience=args.patience)


# -----------------------------
# Training loop
# -----------------------------
best = None

# Init Logger

logger = ContinuousLogger(Path('results_run_joint_classifier_guidance'), 'run_joint_classifier_guidance', 'train')

for epoch in range(1, args.epochs + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        l_diff = model.diffusion_loss(x)
        l_clf = model.classifier_loss(x, y)
        loss = l_diff + 0.3 * l_clf
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Validation
    model.eval()
    v, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            v += (model.diffusion_loss(x) + 0.3 * model.classifier_loss(x, y)).item() * x.size(0)
            n += x.size(0)
    v /= n
    # Log

    msg = f'Epoch {epoch}: val_total={v:.4f}'

    logger.log_console(msg)

    logger.log_epoch_stats({

        "epoch": epoch,

        "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

        "train_loss": loss.item() if 'loss' in locals() else 0

    })

    if stop.step(v):
        best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if stop.should_stop():
        break


# Load best model
if best is not None:
    model.load_state_dict(best)


# -----------------------------
# Sampling with classifier guidance
# -----------------------------
os.makedirs(args.save, exist_ok=True)
with torch.no_grad():
    # Example: 6 samples per class for CIFAR-10
    y = torch.arange(10, device=device).repeat_interleave(6)[:60]
    imgs = model.sample(B=y.size(0), y=y, guidance=args.guidance, steps=args.steps, device=device)
    grid = make_grid((imgs + 1) / 2, nrow=10)
    save_samples(grid, os.path.join(args.save, 'samples_jcg.png'))
    torch.save(model.state_dict(), os.path.join(args.save, 'jcg.pth'))
