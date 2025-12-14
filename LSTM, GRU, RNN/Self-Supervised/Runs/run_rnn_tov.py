import argparse
import random
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from rnn_tov import TOVGRU

class ToyTOV(Dataset):
    def __init__(self, n=40000, T=64, D=16, num_chunks=4, seed=42):
        rng = np.random.RandomState(seed)
        self.X = []
        self.y = []
        C = num_chunks
        for _ in range(n):
            base = rng.randn(T, D).astype(np.float32)
            base = np.cumsum(base, axis=0) * 0.05
            if rng.rand() < 0.5:
                self.X.append(base)
                self.y.append(1)
            else:
                chunks = np.array_split(base, C, axis=0)
                rng.shuffle(chunks)
                self.X.append(np.concatenate(chunks, axis=0))
                self.y.append(0)
        self.X = np.stack(self.X, axis=0)
        self.y = np.array(self.y, dtype=np.int64)
        self.num_chunks = num_chunks
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience=patience; self.min_delta=min_delta; self.best=float('inf'); self.count=0
    def step(self, v):
        if v < self.best - self.min_delta: self.best=v; self.count=0; return True
        self.count+=1; return False
    def should_stop(self): return self.count>=self.patience


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--T', type=int, default=64)
    ap.add_argument('--D', type=int, default=16)
    ap.add_argument('--chunks', type=int, default=4)
    ap.add_argument('--n', type=int, default=50000)
    ap.add_argument('--val_split', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save', type=str, default='tov_gru_best.pt')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    ds = ToyTOV(args.n, args.T, args.D, args.chunks, args.seed)
    n_val = int(len(ds)*args.val_split); n_tr = len(ds)-n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(args.seed))

    tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    va = DataLoader(va_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TOVGRU(args.D, args.hidden, args.layers, args.chunks).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    es = EarlyStopper(args.patience, 1e-4)

    best=None

    # Init Logger

    logger = ContinuousLogger(Path('results_run_rnn_tov'), 'run_rnn_tov', 'train')

    for epoch in range(1, args.epochs+1):
        net.train(); tr_loss=0.0
        for x,y in tr:
            x=x.to(device); y=y.to(device)
            logit = net(x)
            loss = F.cross_entropy(logit, y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(tr.dataset)

        net.eval(); va_loss=0.0
        with torch.no_grad():
            for x,y in va:
                x=x.to(device); y=y.to(device)
                logit = net(x)
                loss = F.cross_entropy(logit, y)
                va_loss += loss.item() * x.size(0)
        va_loss /= len(va.dataset)

        improved = es.step(va_loss)
        if improved: best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        # Log

        msg = f"Epoch {epoch:03d} | train {tr_loss:.6f} | val {va_loss:.6f} | best {es.best:.6f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if es.should_stop(): print('Early stopping.'); break

    if best is not None: net.load_state_dict(best)
    torch.save(net.state_dict(), args.save)
    print(f"Saved best model to {args.save}")

if __name__ == '__main__':
    main()