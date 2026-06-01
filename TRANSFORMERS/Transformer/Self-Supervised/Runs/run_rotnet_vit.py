"""Runner: RotNet-ViT (predict 0/90/180/270)
Self-supervised pretext framed as 4-class classification.
"""
import argparse
import random

import torch
import torch.optim as optim
from torchvision import datasets, transforms

from rotnet_vit import RotNetConfig, RotNetViT
from TRANSFORMERS.Transformer.Supervised.Runs._common_real_image import make_real_image_loaders


class RotNetDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, dims=[1, 2])
        return img, k


def train_epoch(model, loader, device, opt):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        loss, _ = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        total += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
    return total / len(loader.dataset), correct / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ratio", type=float, default=4.0)
    ap.add_argument("--save", default="rotnet_vit_best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_real_image_loaders(
        args.data_dir, batch_size=args.batch_size, image_size=args.img
    )
    train_loader = torch.utils.data.DataLoader(RotNetDataset(train_loader.dataset), batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(RotNetDataset(val_loader.dataset), batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(RotNetDataset(test_loader.dataset), batch_size=args.batch_size, shuffle=False)

    cfg = RotNetConfig(args.img, args.patch, args.dim, args.depth, args.heads, args.ratio, 4)
    model = RotNetViT(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best = float("inf")
    best_state = None
    bad = 0
    for ep in range(1, args.epochs + 1):
        trl = train_epoch(model, train_loader, device, opt)
        val, acc = eval_epoch(model, val_loader, device)
        print(f"epoch {ep:03d} | train {trl:.4f} | val {val:.4f} | val_acc {acc:.3f}")
        if val + 1e-6 < best:
            best = val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break
    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, args.save)
        print(f"Saved {args.save} (val {best:.4f})")
    test, acc = eval_epoch(model, test_loader, device)
    print(f"Test pretext loss: {test:.4f} | acc {acc:.3f}")


if __name__ == "__main__":
    main()
