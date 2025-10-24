# runs/run_edm.py
import os
import argparse
import torch
from torchvision.utils import make_grid
from models.edm_model_min import EDM
from runs._common_train_c import make_cifar10_loaders, EarlyStopper, save_samples


# -------------------------------------------------------
# Argument parsing
# -------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--patience', type=int, default=20)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--device', type=str, default='cuda')
p.add_argument('--save', type=str, default='./results/edm')
p.add_argument('--steps', type=int, default=40)
args = p.parse_args([]) if __name__ == '__main__' else p.parse_args()


# -------------------------------------------------------
# Data loaders
# -------------------------------------------------------
train_loader, val_loader, _ = make_cifar10_loaders(batch_size=args.batch_size)


# -------------------------------------------------------
# Model setup
# -------------------------------------------------------
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = EDM().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
stop = EarlyStopper(patience=args.patience)


# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
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
    v = 0.0
    n = 0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            v += model.loss(x).item() * x.size(0)
            n += x.size(0)
    v /= n

    print(f'Epoch {epoch}: val_loss={v:.4f}')

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
    # Euler sampler
    euler = model.sample_euler(B=64, steps=args.steps, device=device)
    grid1 = make_grid((euler + 1) / 2, nrow=8)
    save_samples(grid1, os.path.join(args.save, 'samples_edm_euler.png'))

    # Heun sampler
    heun = model.sample_heun(B=64, steps=args.steps, device=device)
    grid2 = make_grid((heun + 1) / 2, nrow=8)
    save_samples(grid2, os.path.join(args.save, 'samples_edm_heun.png'))

# Save final model weights
torch.save(model.state_dict(), os.path.join(args.save, 'edm.pth'))
