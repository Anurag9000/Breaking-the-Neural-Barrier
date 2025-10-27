import argparse, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from ADP_Informer_model import build_informer
from ADP_RetNet_model import TrainCfg, ADPCfg, adp_search, evaluate


def make_loaders(data_root, batch_size, val_split=5000, download=True):
    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    te_tf = transforms.ToTensor()
    full = datasets.CIFAR10(root=data_root, train=True, transform=tr_tf, download=download)
    te = datasets.CIFAR10(root=data_root, train=False, transform=te_tf, download=download)
    train_len = len(full)-val_split
    train_ds, val_ds = random_split(full, [train_len, val_split], generator=torch.Generator().manual_seed(42))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
            DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='./data')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--val-split', type=int, default=5000)
    p.add_argument('--download', action='store_true')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--delta', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)

    p.add_argument('--adp-mode', choices=['width_to_depth','depth_to_width','alt_depth','alt_width','depth_only','width_only'], default='width_to_depth')
    p.add_argument('--init-width', type=int, default=64)
    p.add_argument('--init-depth', type=int, default=2)
    p.add_argument('--ex-k', type=int, default=16)
    p.add_argument('--max-width', type=int, default=256)
    p.add_argument('--max-depth', type=int, default=16)
    p.add_argument('--trials-width', type=int, default=2)
    p.add_argument('--trials-depth', type=int, default=2)

    args = p.parse_args()
    train_loader, val_loader, test_loader = make_loaders(args.data_root, args.batch_size, args.val_split, args.download)

    model = build_informer(num_classes=10, init_width=args.init_width, init_depth=args.init_depth)

    train_cfg = TrainCfg(lr=args.lr, wd=args.wd, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, delta=args.delta)
    adp = ADPCfg(init_width=args.init_width, init_depth=args.init_depth, ex_k=args.ex_k, max_width=args.max_width, max_depth=args.max_depth, trials_width=args.trials_width, trials_depth=args.trials_depth)

    val = adp_search(model, train_loader, val_loader, train_cfg, adp, mode=args.adp_mode)
    print({'val_loss': val[0], 'val_acc': val[1]})

    from ADP_RetNet_model import evaluate as eval_fn
    test_loss, test_acc = eval_fn(model, test_loader, train_cfg.device)
    print({'test_loss': test_loss, 'test_acc': test_acc})

if __name__ == '__main__':
    main()
