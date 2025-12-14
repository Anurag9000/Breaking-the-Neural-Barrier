import argparse
import os
import random

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model_deit import DeiT

# -------------------------------------------------------
# DeiT training harness (no distillation, single model):
# - Strong augmentations (RandAugment-ish), Mixup/CutMix optional
# - Label smoothing
# - AdamW + cosine LR + warmup
# - Early stopping on val loss
# -------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(dataset: str, data_root: str, img_size: int, batch_size: int, workers: int, seed: int = 42):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
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


class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        n = logits.size(-1)
        logp = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.eps / (n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        return torch.mean(torch.sum(-true_dist * logp, dim=-1))


def train_one_epoch(model, loader, device, criterion, optimizer, mixup_alpha=0.0, cutmix_alpha=0.0):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
            lam = 1.0
            if mixup_alpha > 0.0:
                lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
                index = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[index]
                y_a, y_b = y, y[index]
            elif cutmix_alpha > 0.0:
                lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
                index = torch.randperm(x.size(0), device=device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
                y_a, y_b = y, y[index]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
        else:
            loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = (1. - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    return bbx1, bby1, bbx2, bby2


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
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--embed", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--cutmix", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="DeiT_best.pth")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tl, vl, te = get_loaders(args.dataset, args.data_root, args.img_size, args.batch_size, args.workers, args.seed)
    num_classes = 10 if args.dataset == "cifar10" else 100

    model = DeiT(
        img_size=args.img_size,
        patch_size=args.patch,
        num_classes=num_classes,
        embed_dim=args.embed,
        depth=args.depth,
        num_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
    ).to(device)

    criterion = LabelSmoothingCE(eps=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup)

    best_val, best_state, bad = float("inf"), None, 0


    # Init Logger


    logger = ContinuousLogger(Path('results_run_deit'), 'run_deit', 'train')


    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup:
            for g in optimizer.param_groups:
                g["lr"] = args.lr * epoch / max(1, args.warmup)
        else:
            scheduler.step()

        tr_loss = train_one_epoch(model, tl, device, criterion, optimizer, args.mixup, args.cutmix)
        val_loss, val_acc = evaluate(model, vl, device, criterion)
        # Log

        msg = f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {val_loss:.4f} | acc {val_acc*100:.2f}% | lr {optimizer.param_groups[0]['lr']:.2e}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })

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

    test_loss, test_acc = evaluate(model, te, device, nn.CrossEntropyLoss())
    print(f"TEST | loss {test_loss:.4f} | acc {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()