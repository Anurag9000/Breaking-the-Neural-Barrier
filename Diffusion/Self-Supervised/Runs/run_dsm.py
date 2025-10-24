# runs/run_dsm.py
import os
import argparse
import torch
from torchvision.utils import make_grid
from models.dsm_model import DSM
from runs._common_train_b import make_cifar10_loaders, EarlyStopper, save_samples


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--sigma', type=float, default=0.2)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--save', type=str, default='./results/dsm')
args = parser.parse_args([]) if __name__ == '__main__' else parser.parse_args()


# Data loaders
train_loader, val_loader, _ = make_cifar10_loaders(batch_size=args.batch_size)

# Device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Model and optimizer
model = DSM(sigma=args.sigma).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
stop = EarlyStopper(patience=args.patience)

best_state = None

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

    print(f'Epoch {epoch}: val_loss={vloss:.4f}')

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
    samples = model.langevin_sample(B=64, steps=400, step_size=0.01, device=device)
    grid = make_grid((samples + 1) / 2, nrow=8)
    save_samples(grid, os.path.join(args.save, 'samples_dsm.png'))
    torch.save(model.state_dict(), os.path.join(args.save, 'dsm.pth'))
