import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torchvision as tv
import torchvision.transforms as T

from adp_cnn_alt_depth import (
    AdaptiveCNN,
    InnerCfg,
    SearchCfg,
    alternating_adp_search_depth_first,
)


def make_loaders(data_root: str, batch_size: int, num_workers: int, seed: int):
    normalize = T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_tf = T.Compose([T.ToTensor(), normalize])

    train_full = tv.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test = tv.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    g = torch.Generator().manual_seed(seed)
    n_train = int(0.9 * len(train_full))
    n_val = len(train_full) - n_train
    train, val = random_split(train_full, [n_train, n_val], generator=g)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    return train_loader, val_loader, test_loader


def eval_top1(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    import torch.nn.functional as F
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(1, total)


def main():
    p = argparse.ArgumentParser()
    # data & system
    p.add_argument('--data-root', type=str, default='./data')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)

    # model seed
    p.add_argument('--seed-width', type=int, default=1)
    p.add_argument('--seed-depth', type=int, default=2)
    p.add_argument('--pool-idx', type=int, nargs='*', default=[0, 2])

    # inner training
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--max-epochs-inner', type=int, default=200)
    p.add_argument('--es-patience', type=int, default=10)
    p.add_argument('--grad-clip', type=float, default=1.0)

    # search
    p.add_argument('--delta', type=float, default=1e-3)
    p.add_argument('--patience-width', type=int, default=3)
    p.add_argument('--patience-depth', type=int, default=3)
    p.add_argument('--max-total-epochs', type=int, default=5000)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = make_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    widths = [args.seed_width] * args.seed_depth
    model = AdaptiveCNN(in_ch=3, num_classes=10, widths=widths, pooling_indices=args.pool-idx)

    inner_cfg = InnerCfg(
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs_inner=args.max_epochs_inner,
        es_patience=args.es_patience,
        grad_clip=args.grad_clip,
    )
    search_cfg = SearchCfg(
        delta=args.delta,
        patience_width=args.patience_width,
        patience_depth=args.patience_depth,
        max_total_epochs=args.max_total_epochs,
    )

    model, best_val, spent = alternating_adp_search_depth_first(
        model, train_loader, val_loader, device, inner_cfg, search_cfg
    )

    top1 = eval_top1(model, test_loader, device)

    print('\n===== RESULTS (Depth-first) =====')
    print(f'Best Val Loss   : {best_val:.4f}')
    print(f'Total Epochs    : {spent}')
    print(f'Final Depth     : {len(model.widths)}')
    print(f'Final Widths    : {model.widths}')
    print(f'Test@1 Accuracy : {top1:.2f}%')

if __name__ == '__main__':
    main()
