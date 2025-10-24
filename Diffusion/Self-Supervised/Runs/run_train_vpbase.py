import os
import argparse
import torch
from torchvision.utils import make_grid
from models.i_base_vp import VPBase
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
p.add_argument('--save', type=str, default='./results/partI_base')
p.add_argument('--steps', type=int, default=50)
args = p.parse_args([]) if __name__ == '__main__' else p.parse_args()


# -----------------------------
# Data loaders
# -----------------------------
train_loader, val_loader, _ = make_cifar10_loaders(batch_size=args.batch_size)


# -----------------------------
# Model and optimizer
# -----------------------------
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = VPBase().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
stop = EarlyStopper(patience=args.patience)


# -----------------------------
# Training loop
# -----------------------------
best = None

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
    val_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            val_loss += model.loss(x).item() * x.size(0)
            n += x.size(0)
    val_loss /= n
    print(f'Epoch {epoch}: val_loss={val_loss:.4f}')

    # Early stopping
    if stop.step(val_loss):
        # Save best model state
        best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if stop.should_stop():
        print("Early stopping triggered.")
        break

# Load best model if available
if best is not None:
    model.load_state_dict(best)


# -----------------------------
# Save model and generate samples
# -----------------------------
os.makedirs(args.save, exist_ok=True)

# Quick DDIM sampling (unguided)
with torch.no_grad():
    x = torch.randn(64, 3, 32, 32, device=device)
    for i in reversed(range(1, args.steps + 1)):
        t = torch.full((64,), int(model.T * i / args.steps), device=device, dtype=torch.long)
        t_prev = torch.full((64,), int(model.T * (i - 1) / args.steps), device=device, dtype=torch.long)
        eps_hat = model.net(x, t.float() / model.T)
        x = model.ddim_step(x, t, t_prev, eps_hat)

    grid = make_grid((x.clamp(-1, 1) + 1) / 2, nrow=8)
    save_samples(grid, os.path.join(args.save, 'samples_base.png'))

# Save model weights
torch.save(model.state_dict(), os.path.join(args.save, 'vp_base.pth'))
