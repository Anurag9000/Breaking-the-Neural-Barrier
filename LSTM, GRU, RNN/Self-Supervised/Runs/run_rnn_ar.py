import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from rnn_ar import ARGRU

class ToyARSeq(Dataset):
    def __init__(self, n=50000, T=64, D=16, seed=42):
        rng = np.random.RandomState(seed)
        self.X = []
        for _ in range(n):
            base = rng.randn(T, D).astype(np.float32)
            base = np.cumsum(base, axis=0) * 0.1  # random walk
            self.X.append(base)
        self.X = np.stack(self.X, axis=0)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.count = 0
    def step(self, v):
        if v < self.best - self.min_delta:
            self.best = v; self.count = 0; return True
        self.count += 1; return False
    def should_stop(self):
        return self.count >= self.patience


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--T', type=int, default=64)
    ap.add_argument('--D', type=int, default=16)
    ap.add_argument('--n', type=int, default=60000)
    ap.add_argument('--val_split', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save', type=str, default='ar_gru_best.pt')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    ds = ToyARSeq(args.n, args.T, args.D, args.seed)
    n_val = int(len(ds) * args.val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ARGRU(args.D, args.hidden_dim, args.layers).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    es = EarlyStopper(args.patience, 1e-4)

    best = None
    for epoch in range(1, args.epochs+1):
        net.train(); tr = 0.0
        for x in train_loader:
            x = x.to(device)
            y = x[:,1:,:]
            inp = x[:,:-1,:]
            pred = net(inp)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tr += loss.item() * x.size(0)
        tr /= len(train_loader.dataset)

        net.eval(); va = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                y = x[:,1:,:]; inp = x[:,:-1,:]
                pred = net(inp)
                loss = loss_fn(pred, y)
                va += loss.item() * x.size(0)
        va /= len(val_loader.dataset)

        improved = es.step(va)
        if improved:
            best = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
        print(f"Epoch {epoch:03d} | train {tr:.6f} | val {va:.6f} | best {es.best:.6f}")
        if es.should_stop():
            print('Early stopping.'); break

    if best is not None:
        net.load_state_dict(best)
    torch.save(net.state_dict(), args.save)
    print(f"Saved best model to {args.save}")

if __name__ == '__main__':
    main()
