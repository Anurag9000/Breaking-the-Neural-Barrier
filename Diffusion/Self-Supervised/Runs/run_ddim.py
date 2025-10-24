# runs/run_ddim.py
import os
import argparse
import torch
from torchvision.utils import make_grid
from models.ddim_model import DDIMSampler
from models.ddpm_eps_model import DDPMEps
from runs._common_train import make_cifar10_loaders, EarlyStopper, save_samples


# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--eta', type=float, default=0.0)
parser.add_argument('--schedule', type=str, default='linear', choices=['linear', 'cosine'])
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--save', type=str, default='./results/ddim')
parser.add_argument('--ckpt', type=str, default='')

args = parser.parse_args([]) if __name__ == '__main__' else parser.parse_args()


# -------------------------
# Data
# -------------------------
train_loader, val_loader, _ = make_cifar10_loaders(batch_size=args.batch_size)


# -------------------------
# Device & model
# -------------------------
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = DDPMEps(T=args.T, schedule=args.schedule).to(device)


# -------------------------
# Load checkpoint or train
# -------------------------
if args.ckpt and os.path.isfile(args.ckpt):
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
else:
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
# DDIM Sampling
# -------------------------
sampler = DDIMSampler(model, T=args.T, schedule=args.schedule)
os.makedirs(args.save, exist_ok=True)

model.eval()
with torch.no_grad():
    samples = sampler.sample(B=64, steps=args.steps, eta=args.eta, device=device)
    grid = make_grid((samples + 1) / 2, nrow=8)  # scale [-1,1] → [0,1]
    save_samples(grid, os.path.join(args.save, 'samples_ddim.png'))
    torch.save(model.state_dict(), os.path.join(args.save, 'ddpm_eps_for_ddim.pth'))
