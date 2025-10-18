import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision as tv
import torchvision.transforms as T

from AE_SPARSE_STL import AE_SPARSE_STL, sparsity_penalty, ae_sparse_total_neurons

# -----------------------------------------------------------------------------
# Runner for AE_SPARSE_STL
# Adds sparsity regularization (L1 or KL) on latent activations
# -----------------------------------------------------------------------------

def build_dataloaders(dataset, data_dir, batch_size, num_workers, val_frac, seed):
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


def train_one_epoch(model, loader, opt, device, sparse_lambda, sparse_mode, rho):
    model.train()
    mse = nn.MSELoss()
    total_loss, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        x_rec, z = model(x)
        loss = mse(x_rec, x)
        if sparse_lambda > 0:
            sp = sparsity_penalty(z, mode=sparse_mode, rho=rho)
            loss = loss + sparse_lambda * sp
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def eval_epoch(model, loader, device, sparse_lambda, sparse_mode, rho):
    model.eval()
    mse = nn.MSELoss(reduction='sum')
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            x_rec, z = model(x)
            loss = mse(x_rec, x)
            if sparse_lambda > 0:
                sp = sparsity_penalty(z, mode=sparse_mode, rho=rho)
                loss = loss + sparse_lambda * sp
            total_loss += float(loss.item())
            n += x.size(0)
    return total_loss / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--dataset', type=str, default='cifar10')
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--out_dir', type=str, default='./runs/ae_sparse_stl')
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
    # Sparsity
    p.add_argument('--sparse_lambda', type=float, default=1e-4, help='scale of sparsity penalty')
    p.add_argument('--sparse_mode', type=str, default='l1', choices=['l1','kl'])
    p.add_argument('--rho', type=float, default=0.05, help='target sparsity for KL')

    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    pool_after = [] if args.pool_after.strip()=='' else [int(x) for x in args.pool_after.split(',')]

    train_loader, val_loader, test_loader = build_dataloaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.val_frac, args.seed)

    model = AE_SPARSE_STL(in_channels=3, width=args.width, depth=args.depth, pool_after=pool_after).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_epoch, epochs_no_improve = float('inf'), -1, 0
    ckpt_path = Path(args.out_dir) / 'best.pt'
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device, args.sparse_lambda, args.sparse_mode, args.rho)
        val_loss = eval_epoch(model, val_loader, device, args.sparse_lambda, args.sparse_mode, args.rho)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val, best_epoch, epochs_no_improve = val_loss, epoch, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, ckpt_path)
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | best={best_val:.4f} @ {best_epoch}")
        if epochs_no_improve >= args.patience:
            print(f"Early stopping at {epoch}")
            break

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    test_loss = eval_epoch(model, test_loader, device, 0.0, args.sparse_mode, args.rho)

    report = {
        'dataset': args.dataset,
        'width': args.width,
        'depth': args.depth,
        'pool_after': pool_after,
        'sparse_mode': args.sparse_mode,
        'sparse_lambda': args.sparse_lambda,
        'rho': args.rho,
        'neurons_metric': ae_sparse_total_neurons(args.width, args.depth),
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
        'test_loss': test_loss,
    }

    with open(Path(args.out_dir)/'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
