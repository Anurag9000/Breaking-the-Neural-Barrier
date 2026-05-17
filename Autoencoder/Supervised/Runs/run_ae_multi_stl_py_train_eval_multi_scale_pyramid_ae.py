import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from _common_real_image import make_real_image_loaders

from AE_MULTI_STL import AE_MULTI_STL, ae_multi_total_neurons

# -----------------------------------------------------------------------------
# Runner for AE_MULTI_STL (multi-scale / pyramid AE)
# - Produces reconstructions at multiple decoder stages (coarse->fine).
# - Loss = sum_i w_i * MSE(pred_i, downsample(target, size_i)).
# - Early stops on the total validation loss.
# -----------------------------------------------------------------------------

def build_dataloaders(dataset: str, data_dir: str, batch_size: int, num_workers: int, val_frac: float, seed: int):
    del dataset, seed
    return make_real_image_loaders(data_root=data_dir, batch_size=batch_size, val_ratio=val_frac, num_workers=num_workers, image_size=224)


def _downsample_to(x: torch.Tensor, y_like: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.interpolate(x, size=y_like.shape[-2:], mode='bilinear', align_corners=False)


def _parse_weights(s: str, depth: int) -> List[float]:
    if s.strip() == '':
        # default: geometric weights favoring finer scales
        # create weights proportional to 2^{i} for i=0..depth-1
        ws = [2.0 ** i for i in range(depth)]
    else:
        ws = [float(x) for x in s.split(',')]
        if len(ws) != depth:
            raise ValueError(f"Expected {depth} weights, got {len(ws)}")
    # normalize to sum=1 for stability
    ssum = sum(ws)
    return [w / ssum for w in ws]


def multi_scale_loss(preds: List[torch.Tensor], target: torch.Tensor, weights: List[float]) -> torch.Tensor:
    # preds are ordered coarse->fine (len == depth)
    assert len(preds) == len(weights)
    mse = nn.MSELoss()
    total = 0.0
    for p, w in zip(preds, weights):
        y = _downsample_to(target, p)
        total = total + w * mse(p, y)
    return total


def train_one_epoch(model, loader, opt, device, weights, scaler=None):
    model.train()
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if scaler is None:
            preds, _ = model(x)
            loss = multi_scale_loss(preds, x, weights)
            loss.backward()
            opt.step()
        else:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cuda' else torch.float16):
                preds, _ = model(x)
                loss = multi_scale_loss(preds, x, weights)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def eval_epoch(model, loader, device, weights):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            preds, _ = model(x)
            loss = multi_scale_loss(preds, x, weights)
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    # Data/IO
    p.add_argument('--dataset', type=str, default='imagefolder', choices=['imagefolder'])
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--out_dir', type=str, default='./Runs/ae_multi_stl')
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
    # Pyramid loss
    p.add_argument('--pyr_weights', type=str, default='', help='comma weights coarse->fine; blank = geometric then normalized')

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

    model = AE_MULTI_STL(in_channels=3, width=args.width, depth=args.depth, pool_after=pool_after).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=='cuda')

    # derive weights for #outputs == depth
    weights = _parse_weights(args.pyr_weights, args.depth)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    ckpt_path = out_dir / 'best.pt'

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, weights, scaler if args.amp else None)
        val_loss = eval_epoch(model, val_loader, device, weights)

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

    # Evaluate: also report full-res MSE using last (finest) head only
    model.eval()
    mse = nn.MSELoss(reduction='sum')
    total_full, total_multi, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device, non_blocking=True)
            preds, _ = model(x)
            total_multi += float(multi_scale_loss(preds, x, weights).item()) * x.size(0)
            # full-res is last prediction
            full = preds[-1]
            y = full  # already full-res
            total_full += float(mse(full, x).item())
            n += x.size(0)
    test_multi = total_multi / max(n, 1)
    test_full = total_full / max(n, 1)

    report = {
        'dataset': args.dataset,
        'width': args.width,
        'depth': args.depth,
        'pool_after': pool_after,
        'neurons_metric': ae_multi_total_neurons(args.width, args.depth),
        'pyr_weights': weights,
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
        'test_multi': test_multi,
        'test_full_mse': test_full,
    }
    with open(out_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
