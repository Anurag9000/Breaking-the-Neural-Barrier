import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from _common_real_image import make_real_image_loaders

from AE_ROBUST_STL import AE_ROBUST_STL, ae_robust_total_neurons

# -----------------------------------------------------------------------------
# Runner for AE_ROBUST_STL using Huber (SmoothL1) reconstruction loss.
# Early stopping on validation Huber objective; test reports Huber and MSE.
# -----------------------------------------------------------------------------

def build_dataloaders(dataset: str, data_dir: str, batch_size: int, num_workers: int, val_frac: float, seed: int):
    del dataset, seed
    return make_real_image_loaders(data_root=data_dir, batch_size=batch_size, val_ratio=val_frac, num_workers=num_workers, image_size=224)


def train_one_epoch(model, loader, opt, device, delta: float, scaler=None):
    model.train()
    huber = nn.SmoothL1Loss(beta=delta)
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if scaler is None:
            x_rec, _ = model(x)
            loss = huber(x_rec, x)
            loss.backward()
            opt.step()
        else:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cuda' else torch.float16):
                x_rec, _ = model(x)
                loss = huber(x_rec, x)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def eval_epoch(model, loader, device, delta: float):
    model.eval()
    huber = nn.SmoothL1Loss(beta=delta, reduction='sum')
    mse = nn.MSELoss(reduction='sum')
    total_huber, total_mse, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            x_rec, _ = model(x)
            total_huber += float(huber(x_rec, x).item())
            total_mse += float(mse(x_rec, x).item())
            n += x.size(0)
    return (total_huber / max(n, 1), total_mse / max(n, 1))


def main():
    p = argparse.ArgumentParser()
    # Data/IO
    p.add_argument('--dataset', type=str, default='imagefolder', choices=['imagefolder'])
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--out_dir', type=str, default='./Runs/ae_robust_stl')
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
    # Robust loss
    p.add_argument('--delta', type=float, default=0.1, help='Huber transition (SmoothL1 beta)')

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

    model = AE_ROBUST_STL(in_channels=3, width=args.width, depth=args.depth, pool_after=pool_after).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=='cuda')

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    ckpt_path = out_dir / 'best.pt'

    for epoch in range(1, args.epochs + 1):
        train_huber = train_one_epoch(model, train_loader, optimizer, device, args.delta, scaler if args.amp else None)
        val_huber, _ = eval_epoch(model, val_loader, device, args.delta)

        improved = val_huber < best_val - 1e-6
        if improved:
            best_val = val_huber
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_huber': val_huber, 'args': vars(args)}, ckpt_path)
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:03d} | train_huber={train_huber:.4f} | val_huber={val_huber:.4f} | best_val={best_val:.4f} @ {best_epoch}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Load best and evaluate on test (both Huber & MSE for reference)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model'])
    test_huber, test_mse = eval_epoch(model, test_loader, device, args.delta)

    report = {
        'dataset': args.dataset,
        'width': args.width,
        'depth': args.depth,
        'pool_after': pool_after,
        'neurons_metric': ae_robust_total_neurons(args.width, args.depth),
        'delta': args.delta,
        'best_val_huber': best_val,
        'best_epoch': best_epoch,
        'test_huber': test_huber,
        'test_mse': test_mse,
    }
    with open(out_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
