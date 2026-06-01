import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from _common_forda import make_forda_sequence_loaders
from rnn_ar_py_autoregressive_next_step_modeling_gru_rnn import ARGRU


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
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default="ar_gru_best.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, val_loader, _, _ = make_forda_sequence_loaders(batch_size=args.batch_size, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ARGRU(1, args.hidden_dim, args.layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    es = EarlyStopper(args.patience, 1e-4)

    best = None
    for epoch in range(1, args.epochs + 1):
        net.train()
        train_loss = 0.0
        for x in train_loader:
            x = _to_sequence(x).to(device)
            pred = net(x[:, :-1, :])
            loss = loss_fn(pred, x[:, 1:, :])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = _to_sequence(x).to(device)
                pred = net(x[:, :-1, :])
                loss = loss_fn(pred, x[:, 1:, :])
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | best {es.best:.6f}")
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
