import argparse
import math
import random
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.time_series_benchmarks import make_forda_sequence_loaders

from rnn_mtm import MaskedTimeModel


def make_mask(B, T, mask_prob, device):
    m = torch.rand(B, T, device=device) < mask_prob
    return m

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.count = 0
    def step(self, val):
        if val < self.best - self.min_delta:
            self.best = val
            self.count = 0
            return True
        else:
            self.count += 1
            return False
    def should_stop(self):
        return self.count >= self.patience


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--mask_prob', type=float, default=0.15)
    ap.add_argument('--T', type=int, default=64)
    ap.add_argument('--D', type=int, default=16)
    ap.add_argument('--n_samples', type=int, default=30000)
    ap.add_argument('--val_split', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save', type=str, default='mtm_bigru_best.pt')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_base, val_base, _, _ = make_forda_sequence_loaders(batch_size=args.batch_size, seed=args.seed, return_index=False)
    train_loader = DataLoader(train_base.dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_base.dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MaskedTimeModel(1, args.hidden_dim, args.layers).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction='sum')
    es = EarlyStopper(patience=args.patience, min_delta=1e-4)

    best = None

    # Init Logger

    logger = ContinuousLogger(Path('results_run_rnn_mtm'), 'run_rnn_mtm', 'train')

    for epoch in range(1, args.epochs+1):
        net.train()
        tr = 0.0
        ntr = 0
        for x in train_loader:
            x = x.transpose(1, 2).to(device)
            B,T,D = x.shape
            mask = make_mask(B,T,args.mask_prob,device)
            x_masked = x.clone()
            x_masked[mask] = 0.0  # simple zero-mask
            y = net(x_masked)
            # compute loss only on masked positions
            diff = (y - x)
            diff = diff[mask]
            loss = (diff**2).sum() / (mask.sum().clamp(min=1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            tr += loss.item() * B
            ntr += B
        tr /= ntr

        net.eval()
        va = 0.0
        nva = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.transpose(1, 2).to(device)
                B,T,D = x.shape
                mask = make_mask(B,T,args.mask_prob,device)
                x_masked = x.clone(); x_masked[mask] = 0.0
                y = net(x_masked)
                diff = (y - x)[mask]
                loss = (diff**2).sum() / (mask.sum().clamp(min=1))
                va += loss.item() * B
                nva += B
        va /= nva

        improved = es.step(va)
        if improved:
            best = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
        # Log

        msg = f"Epoch {epoch:03d} | train {tr:.6f} | val {va:.6f} | best {es.best:.6f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if es.should_stop():
            print('Early stopping.')
            break

    if best is not None:
        net.load_state_dict(best)
    torch.save(net.state_dict(), args.save)
    print(f"Saved best model to {args.save}")

if __name__ == '__main__':
    main()
