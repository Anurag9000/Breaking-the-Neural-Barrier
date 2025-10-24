import os
import argparse
import torch
from torchvision.utils import make_grid
from models.inpaint_ddpm import InpaintDDPM
from runs._common_train_h import make_cifar10_loaders, EarlyStopper, save_samples, random_box_mask

# ---------------------------
# Arguments
# ---------------------------
p = argparse.ArgumentParser()
p.add_argument('--epochs', type=int, default=180)
p.add_argument('--patience', type=int, default=20)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--device', type=str, default='cuda')
p.add_argument('--save', type=str, default='./results/inpaint')
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
model = InpaintDDPM().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
stop = EarlyStopper(patience=args.patience)

# ---------------------------
# Training Loop
# ---------------------------
best = None
for epoch in range(1, args.epochs + 1):
    model.train()
    for x, _ in train_loader:
        x = x.to(device)
        mask = random_box_mask(x.size(0), x.size(2), x.size(3), device=device)
        loss = model.loss(x, mask)

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
            mask = random_box_mask(x.size(0), x.size(2), x.size(3), device=device)
            val_loss += model.loss(x, mask).item() * x.size(0)
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
    x, _ = next(iter(val_loader))
    x = x.to(device)[:64]
    mask = random_box_mask(x.size(0), x.size(2), x.size(3), device=device)
    x_ctx = x * mask
    samples = model.sample(x_ctx, mask, steps=args.steps)

    # Make grid: context + generated samples
    grid = make_grid(torch.cat([(x_ctx + 1) / 2, (samples + 1) / 2], dim=0), nrow=8)
    save_samples(grid, os.path.join(args.save, 'samples_inpaint.png'))

# Save model checkpoint
torch.save(model.state_dict(), os.path.join(args.save, 'inpaint.pth'))
