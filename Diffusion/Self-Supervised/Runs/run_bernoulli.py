# runs/run_bernoulli.py
import os
import argparse
import torch
from torchvision.utils import make_grid
from models.bernoulli_diffusion import BernoulliDiffusion
from runs._common_train_f import make_cifar10_loaders, EarlyStopper, save_samples

# -------------------------
# Command-line arguments
# -------------------------
p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int, default=160)
p.add_argument('--patience', type=int, default=20)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--device', type=str, default='cuda')
p.add_argument('--save', type=str, default='./results/bernoulli')
p.add_argument('--steps', type=int, default=100)

args = p.parse_args([]) if __name__ == '__main__' else p.parse_args()

# -------------------------
# Data loaders
# -------------------------
train_loader, val_loader, _ = make_cifar10_loaders(batch_size=args.batch_size)

# -------------------------
# Model, optimizer, early stopping
# -------------------------
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = BernoulliDiffusion().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
stop = EarlyStopper(patience=args.patience)

# -------------------------
# Training loop
# -------------------------
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
        best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if stop.should_stop():
        print("Early stopping triggered.")
        break

# Load best model
if best is not None:
    model.load_state_dict(best)

# -------------------------
# Sampling and saving results
# -------------------------
os.makedirs(args.save, exist_ok=True)
with torch.no_grad():
    imgs = model.sample(B=64, steps=args.steps, device=device)
    grid = make_grid((imgs + 1) / 2, nrow=8)  # rescale to [0,1]
    save_samples(grid, os.path.join(args.save, 'samples_bernoulli.png'))
    torch.save(model.state_dict(), os.path.join(args.save, 'bernoulli.pth'))
