import argparse
import math
import os
import random
from pathlib import Path

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from utils.adp_logging import ContinuousLogger
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from _common_real_image import infer_num_classes, make_real_image_loaders

from model_vit import VisionTransformer

# ------------------------------
# Supervised training harness on folder-backed real image datasets:
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
    return make_real_image_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=workers,
        image_size=img_size,
    )


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
    parser.add_argument("--dataset", type=str, default="imagefolder", choices=["imagefolder"])
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

    num_classes = infer_num_classes(train_loader)
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


    # Init Logger


    logger = ContinuousLogger(Path('results_run_vit'), 'run_vit', 'train')


    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        # Log


        msg = f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {val_loss:.4f} | acc {val_acc*100:.2f}%"


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

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"TEST | loss {test_loss:.4f} | acc {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
