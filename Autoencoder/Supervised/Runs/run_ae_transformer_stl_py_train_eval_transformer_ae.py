import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from _common_real_image import make_real_image_loaders

from AE_TRANSFORMER_STL import AE_TRANSFORMER_STL, ae_transformer_total_neurons

# -----------------------------------------------------------------------------
# Runner for AE_TRANSFORMER_STL (ViT-style patch AE)
# Loss = MSE reconstruction on pixels; early stopping.
# -----------------------------------------------------------------------------

def build_dataloaders(dataset: str, data_dir: str, batch_size: int, num_workers: int, val_frac: float, seed: int):
    del dataset, seed
    return make_real_image_loaders(data_root=data_dir, batch_size=batch_size, val_ratio=val_frac, num_workers=num_workers, image_size=224)


def train_one_epoch(model, loader, opt, device, scaler=None):
    model.train(); mse = nn.MSELoss(); total, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if scaler is None:
            x_rec, _ = model(x); loss = mse(x_rec, x); loss.backward(); opt.step()
        else:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cuda' else torch.float16):
                x_rec, _ = model(x); loss = mse(x_rec, x)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        total += float(loss.item()) * x.size(0); n += x.size(0)
    return total / max(n,1)


def eval_epoch(model, loader, device):
    model.eval(); mse = nn.MSELoss(reduction='sum'); total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            x_rec, _ = model(x)
            total += float(mse(x_rec, x).item()); n += x.size(0)
    return total / max(n,1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='imagefolder', choices=['imagefolder'])
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--out_dir', type=str, default='./runs/ae_transformer_stl')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--embed_dim', type=int, default=192)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--patch_size', type=int, default=4)
    p.add_argument('--mlp_ratio', type=float, default=4.0)
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

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset=args.dataset, data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, val_frac=args.val_frac, seed=args.seed,
    )

    model = AE_TRANSFORMER_STL(in_channels=3, embed_dim=args.embed_dim, depth=args.depth,
                               num_heads=args.num_heads, patch_size=args.patch_size,
                               mlp_ratio=args.mlp_ratio).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=='cuda')

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    best_val, best_epoch, epochs_no_improve = float('inf'), -1, 0
    ckpt_path = out_dir / 'best.pt'

    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler if args.amp else None)
        val_loss = eval_epoch(model, val_loader, device)
        if val_loss < best_val - 1e-6:
            best_val, best_epoch, epochs_no_improve = val_loss, epoch, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss, 'args': vars(args)}, ckpt_path)
        else:
            epochs_no_improve += 1
        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | best={best_val:.4f} @ {best_epoch}")
        if epochs_no_improve >= args.patience:
            print('Early stopping'); break

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device); model.load_state_dict(state['model'])
    test_mse = eval_epoch(model, test_loader, device)

    report = {
        'dataset': args.dataset, 'embed_dim': args.embed_dim, 'depth': args.depth,
        'num_heads': args.num_heads, 'patch_size': args.patch_size,
        'neurons_metric': ae_transformer_total_neurons(args.embed_dim, args.depth),
        'best_val_loss': best_val, 'best_epoch': best_epoch, 'test_mse': test_mse,
    }
    with open(out_dir / 'report.json', 'w') as f: json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
