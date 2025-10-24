# runs/run_improved_ddpm.py
import os
import argparse
import torch
from torchvision.utils import make_grid
from models.improved_ddpm_model import ImprovedDDPM
from runs._common_train import make_cifar10_loaders, EarlyStopper, save_samples


# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--schedule', type=str, default='cosine', choices=['linear', 'cosine'])
parser.add_argument('--loss_weight', type=str, default='snr', choices=['uniform', 'snr'])
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--save', type=str, default='./results/improved_ddpm')

args = parser.parse_args([]) if __name__ == '__main__' else parser.parse_args()


# -------------------------
# Data
# -------------------------
train_loader, val_loader, _ = make_cifar10_loaders(batch_size=args.batch_size)


# -------------------------
# Device & model
# -------------------------
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = ImprovedDDPM(T=args.T, schedule=args.schedule, loss_weight=args.loss_weight).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
stop = EarlyStopper(patience=args.patience)


# -------------------------
# Training loop
# -------------------------
best_state = None

for epoch in range(1, args.epochs + 1):
    # Train
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
    with torch.no_grad():
        vloss = 0.0
        n = 0
        for x, _ in val_loader:
            x = x.to(device)
            vloss += model.loss(x).item() * x.size(0)
            n += x.size(0)
        vloss /= n

    print(f"Epoch {epoch}: val_loss={vloss:.4f}")

    if stop.step(vloss):
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if stop.should_stop():
        print("Early stopping triggered.")
        break


# Load best model
if best_state is not None:
    model.load_state_dict(best_state)


# -------------------------
# Sampling
# -------------------------
os.makedirs(args.save, exist_ok=True)
model.eval()
with torch.no_grad():
    samples = model.sample(B=64, device=device)
    grid = make_grid((samples + 1) / 2, nrow=8)  # scale [-1,1] → [0,1]
    save_samples(grid, os.path.join(args.save, 'samples_improved_ddpm.png'))
    torch.save(model.state_dict(), os.path.join(args.save, 'improved_ddpm.pth'))
