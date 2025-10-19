import argparse, os, time, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from vae_base import VAE, VAEConfig

# ------------------------------
# Runner: Vanilla VAE on CIFAR-10
# ------------------------------

def get_dataloaders(data_root: str, batch_size: int, num_workers: int = 2):
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
    ])
    full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    n_train = int(0.9*len(full))
    n_val = len(full) - n_train
    train_set, _ = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # Recreate a validation set with eval transform to avoid augment leakage
    full_eval = datasets.CIFAR10(root=data_root, train=True, download=False, transform=tf_eval)
    _, val_set = random_split(full_eval, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

@torch.no_grad()
def evaluate(model: VAE, loader, device):
    model.eval()
    total, total_recon, total_kl = 0.0, 0.0, 0.0
    for x, _ in loader:
        x = x.to(device)
        x_hat, mu, logvar = model(x)
        loss, recon, kl = model.elbo_loss(x, x_hat, mu, logvar)
        bs = x.size(0)
        total += loss.item()*bs
        total_recon += recon.item()*bs
        total_kl += kl.item()*bs
    n = len(loader.dataset)
    return total/n, total_recon/n, total_kl/n


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    train_loader, val_loader = get_dataloaders(args.data, args.batch_size, args.workers)

    cfg = VAEConfig(in_channels=3, latent_dim=args.latent, width=args.width)
    model = VAE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, _, _ = model.elbo_loss(x, x_hat, mu, logvar)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()*x.size(0)
        epoch_loss /= len(train_loader.dataset)

        val_loss, val_recon, val_kl = evaluate(model, val_loader, device)
        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            best_state = { 'model': model.state_dict(), 'cfg': cfg.__dict__, 'epoch': epoch }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.verbose:
            print(f"Epoch {epoch:03d} | train {epoch_loss:.4f} | val {val_loss:.4f} (recon {val_recon:.4f}, kl {val_kl:.4f})")

        if epochs_no_improve >= args.patience:
            if args.verbose:
                print('Early stopping triggered.')
            break

    if best_state is not None:
        os.makedirs(args.out, exist_ok=True)
        torch.save(best_state, os.path.join(args.out, 'vae_base_cifar10.pth'))
        with open(os.path.join(args.out, 'vae_base_metrics.json'), 'w') as f:
            json.dump({'best_val': best_val}, f, indent=2)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--out', type=str, default='./artifacts')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--latent', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    train(args)
