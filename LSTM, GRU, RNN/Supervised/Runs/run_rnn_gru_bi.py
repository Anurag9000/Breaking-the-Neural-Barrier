from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from RNN_GRU_Bi import RNN_GRU_Bi
from _common_real_benchmark import make_caltech256_loaders


def train_eval(model, loaders, device, epochs=40, lr=1e-3, wd=1e-4, patience=7):
    train_loader, val_loader, test_loader, _, _ = loaders
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    best, state, bad = 1e9, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        tot, ok, lsum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                l = ce(logits, y)
                lsum += l.item() * y.size(0)
                ok += (logits.argmax(1) == y).sum().item()
                tot += y.size(0)
        vloss = lsum / tot
        if vloss < best - 1e-4:
            best, state, bad = vloss, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    tot, ok, lsum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            lsum += ce(logits, y).item() * y.size(0)
            ok += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
    print(f"TEST | loss {lsum / tot:.4f} acc {ok / tot:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_caltech256_loaders(root=os.environ.get("BBNB_CALTECH256_ROOT", "./data/Caltech256"), batch_size=args.batch, seed=args.seed)
    _, _, _, input_dim, num_classes = loaders
    model = RNN_GRU_Bi(input_dim=input_dim, hidden_size=args.hidden, num_layers=args.layers, num_classes=num_classes)
    train_eval(model, loaders, device, epochs=args.epochs, lr=args.lr, wd=args.wd)


if __name__ == "__main__":
    main()
