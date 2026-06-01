import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from _common_forda import make_forda_loaders, make_forda_sequence_loaders
from rnn_pl import PLGRU


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.count = 0
            return True
        self.count += 1
        return False

    def should_stop(self) -> bool:
        return self.count >= self.patience


def _to_sequence(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_sup", type=int, default=20)
    ap.add_argument("--epochs_pl", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default="pl_gru_best.pt")
    ap.add_argument("--pl_conf", type=float, default=0.8)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, val_loader, _, num_classes = make_forda_loaders(batch_size=args.batch, seed=args.seed)
    unlab_loader, _, _, _ = make_forda_sequence_loaders(batch_size=args.batch, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PLGRU(1, args.hidden, args.layers, num_classes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    es = EarlyStopper(args.patience, 1e-4)

    for epoch in range(1, args.epochs_sup + 1):
        net.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = _to_sequence(x).to(device)
            y = y.to(device)
            logits = net(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = _to_sequence(x).to(device)
                y = y.to(device)
                logits = net(x)
                loss = F.cross_entropy(logits, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"[Warmup] Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")
        es.step(val_loss)

    best = None
    for epoch in range(1, args.epochs_pl + 1):
        net.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = _to_sequence(x).to(device)
            y = y.to(device)
            logits = net(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * x.size(0)

        for x in unlab_loader:
            x = _to_sequence(x).to(device)
            with torch.no_grad():
                probs = torch.softmax(net(x), dim=-1)
                conf, pseudo = probs.max(dim=-1)
                mask = conf >= args.pl_conf
            if mask.any():
                logits = net(x[mask])
                loss = F.cross_entropy(logits, pseudo[mask])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        train_loss /= len(train_loader.dataset)
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = _to_sequence(x).to(device)
                y = y.to(device)
                logits = net(x)
                loss = F.cross_entropy(logits, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"[PL] Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")
        if es.step(val_loss):
            best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if es.should_stop():
            print("Early stopping.")
            break

    if best is not None:
        net.load_state_dict(best)
    torch.save(net.state_dict(), args.save)
    print(f"Saved best model to {args.save}")


if __name__ == "__main__":
    main()
