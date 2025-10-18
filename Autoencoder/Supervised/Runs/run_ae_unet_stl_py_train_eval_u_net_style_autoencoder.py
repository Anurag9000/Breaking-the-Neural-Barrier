import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision as tv
import torchvision.transforms as T

from AE_UNET_STL import AE_UNET_STL, ae_unet_total_neurons

# -----------------------------------------------------------------------------
# Runner for AE_UNET_STL (U-Net-style AE with encoder-decoder skips)
# Objective: MSE reconstruction; early stopping on validation MSE.
# -----------------------------------------------------------------------------

def build_dataloaders(dataset: str, data_dir: str, batch_size: int, num_workers: int, val_frac: float, seed: int):
    g = torch.Generator().manual_seed(seed)

    if dataset.lower() == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_class = tv.datasets.CIFAR10
    elif dataset.lower() == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ds_class = tv.datasets.CIFAR100
    else:
        raise ValueError('dataset must be cifar10 or cifar100')

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_full = ds_class(root=data_dir, train=True, transform=transform_train, download=True)
    test_ds = ds_class(root=data_dir, train=False, transform=transform_test, download=True)

    val_size = int(len(train_full) * val_frac)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, opt, device, scaler=None):
    model.train()
    mse = nn.MSELoss()
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if scaler is None:
            x_rec, _ = model(x)
            loss = mse(x_rec, x)
            loss.backward()
            opt.step()
        else:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cuda' else torch.float16):
                x_rec, _ = model(x)
                loss = mse(x_rec, x)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def eval_epoch(model, loader, device):
    model.eval()
    mse = nn.MSELoss(reduction='sum')
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            x_rec, _ = model(x)
            total += float(mse(x_rec, x).item())
            n += x.size(0)
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    # Data/IO
    p.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100'])
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--out_dir', type=str, default='./runs/ae_unet_stl')
    p.add_argument('--seed', type=int, default=1337)
    # Model
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--pool_after', type=str, default='2')
    # Train
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--amp', action='store_true')

    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pool_after = [] if args.pool_after.strip()=='' else [int(x) for x in args.pool_after.split(',')]

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    model = AE_UNET_STL(in_channels=3, width=args.width, depth=args.depth, pool_after=pool_after).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=='cuda')

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    ckpt_path = out_dir / 'best.pt'

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler if args.amp else None)
        val_loss = eval_epoch(model, val_loader, device)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss, 'args': vars(args)}, ckpt_path)
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | best_val={best_val:.4f} @ {best_epoch}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model'])
    test_mse = eval_epoch(model, test_loader, device)

    report = {
        'dataset': args.dataset,
        'width': args.width,
        'depth': args.depth,
        'pool_after': pool_after,
        'neurons_metric': ae_unet_total_neurons(args.width, args.depth),
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
        'test_mse': test_mse,
    }
    with open(out_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
