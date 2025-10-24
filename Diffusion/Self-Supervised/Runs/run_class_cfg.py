import os
import argparse
import torch
from torchvision.utils import make_grid
from models.class_cond_cfg_ddpm import ClassCondCFGDDPM
from runs._common_train_h import make_cifar10_loaders, EarlyStopper, save_samples

# ---------------------------
# Arguments
# ---------------------------
p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--patience', type=int, default=20)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--device', type=str, default='cuda')
p.add_argument('--save', type=str, default='./results/class_cfg')
p.add_argument('--steps', type=int, default=50)
p.add_argument('--guidance', type=float, default=1.8)
args = p.parse_args([]) if __name__ == '__main__' else p.parse_args()

# ---------------------------
# Data Loaders
# ---------------------------
train_loader, val_loader, test_loader = make_cifar10_loaders(batch_size=args.batch_size)

# ---------------------------
# Device & Model
# ---------------------------
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = ClassCondCFGDDPM(num_classes=10).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
stop = EarlyStopper(patience=args.patience)

# ---------------------------
# Training Loop
# ---------------------------
best = None
for epoch in range(1, args.epochs + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = model.loss(x, y)

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
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_loss += model.loss(x, y).item() * x.size(0)
            n_samples += x.size(0)
    val_loss /= n_samples
    print(f'Epoch {epoch}: val_loss={val_loss:.4f}')

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
    # Generate 6 samples per class (0-9)
    y = torch.arange(10, device=device).repeat_interleave(6)[:60]
    imgs = model.sample(B=y.size(0), y=y, guidance=args.guidance, steps=args.steps,
                        device=device, size=(3, 32, 32))
    grid = make_grid((imgs + 1) / 2, nrow=10)
    save_samples(grid, os.path.join(args.save, 'samples_class_cfg.png'))

# Save model checkpoint
torch.save(model.state_dict(), os.path.join(args.save, 'class_cfg.pth'))
