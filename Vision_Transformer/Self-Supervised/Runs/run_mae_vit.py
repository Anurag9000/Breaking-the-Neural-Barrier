"""
Runner: MAE-ViT on CIFAR-10/100
- Aug: random resized crop to 32, horizontal flip, color jitter (optional)
- Optim: AdamW
- Early stop on val loss with patience
- Logging to stdout; saves best model
Style mirrors uploaded ADP runners.
"""
import argparse, os, json, time, math, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mae_vit import MAEViT, MAEConfig


def get_data(dataset: str, data_dir: str, batch_size: int, num_workers: int=2):
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465) if dataset=="cifar10" else (0.5071, 0.4867, 0.4408),
        std=(0.2470, 0.2435, 0.2616) if dataset=="cifar10" else (0.2675, 0.2565, 0.2761)
    )
    aug = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    plain = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == "cifar10":
        full_train = datasets.CIFAR10(data_dir, train=True, transform=aug, download=True)
        test = datasets.CIFAR10(data_dir, train=False, transform=plain, download=True)
        n_cls = 10
    else:
        full_train = datasets.CIFAR100(data_dir, train=True, transform=aug, download=True)
        test = datasets.CIFAR100(data_dir, train=False, transform=plain, download=True)
        n_cls = 100
    val_size = 5000
    train_size = len(full_train) - val_size
    train, val = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, device, optimizer, grad_clip=1.0):
    model.train()
    running = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        loss, _ = model(imgs)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    total = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        loss, _ = model(imgs)
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100'])
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--patch_size', type=int, default=4)
    p.add_argument('--embed_dim', type=int, default=384)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--heads', type=int, default=6)
    p.add_argument('--mlp_ratio', type=float, default=4.0)
    p.add_argument('--dec_dim', type=int, default=192)
    p.add_argument('--dec_depth', type=int, default=4)
    p.add_argument('--dec_heads', type=int, default=3)
    p.add_argument('--mask_ratio', type=float, default=0.6)
    p.add_argument('--save', type=str, default='mae_vit_best.pt')
    p.add_argument('--num_workers', type=int, default=2)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_data(args.dataset, args.data_dir, args.batch_size, args.num_workers)

    cfg = MAEConfig(img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
                    depth=args.depth, num_heads=args.heads, mlp_ratio=args.mlp_ratio,
                    decoder_dim=args.dec_dim, decoder_depth=args.dec_depth, decoder_heads=args.dec_heads,
                    mask_ratio=args.mask_ratio)
    model = MAEViT(cfg).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('inf'); best_state=None; patience=args.patience; epochs_no_improve=0
    for epoch in range(1, args.epochs+1):
        tr = train_one_epoch(model, train_loader, device, opt)
        val = eval_loss(model, val_loader, device)
        print(f"epoch {epoch:03d} | train {tr:.4f} | val {val:.4f}")
        if val + 1e-6 < best_val:
            best_val = val; best_state = {k:v.cpu() for k,v in model.state_dict().items()}; epochs_no_improve=0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, args.save)
        print(f"Saved best to {args.save} (val {best_val:.4f})")

    # Report reconstruction loss on test as sanity metric
    test = eval_loss(model, test_loader, device)
    print(f"Test reconstruction loss: {test:.4f}")

if __name__ == '__main__':
    main()
