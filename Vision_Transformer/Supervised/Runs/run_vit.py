import argparse
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model_vit import VisionTransformer

# ------------------------------
# Supervised training harness (CIFAR-10/100) matching your style:
# - Seeded splits (90/10 train/val)
# - Early stopping on val loss with patience
# - AdamW + cosine lr
# - Basic logs and final test eval
# ------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(dataset: str, data_root: str, img_size: int, batch_size: int, workers: int, seed: int = 42):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    ds_cls = datasets.CIFAR10 if dataset.lower() == "cifar10" else datasets.CIFAR100
    full_train = ds_cls(root=data_root, train=True, download=True, transform=train_tf)
    full_eval = ds_cls(root=data_root, train=True, download=True, transform=eval_tf)
    test_set = ds_cls(root=data_root, train=False, download=True, transform=eval_tf)

    n_train = int(0.9 * len(full_train))
    n_val = len(full_train) - n_train
    gen = torch.Generator().manual_seed(seed)
    train_set, _ = random_split(full_train, [n_train, n_val], generator=gen)
    _, val_set = random_split(full_eval, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch", type=int, default=16)
    parser.add_argument("--embed", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="ViT_best.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_loaders(
        args.dataset, args.data_root, args.img_size, args.batch_size, args.workers, args.seed
    )

    num_classes = 10 if args.dataset == "cifar10" else 100
    model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch,
        num_classes=num_classes,
        embed_dim=args.embed,
        depth=args.depth,
        num_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {val_loss:.4f} | acc {val_acc*100:.2f}%")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.save)

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"TEST | loss {test_loss:.4f} | acc {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
