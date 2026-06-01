import argparse
import random
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from utils.adp_logging import ContinuousLogger
from utils.time_series_benchmarks import make_forda_sequence_loaders

from rnn_seqclr import SeqCLRGRU

# simple augmentations for sequences
def augment(x: torch.Tensor, noise_std=0.05, drop_prob=0.1):
    # x: (B,T,D)
    x1 = x + noise_std*torch.randn_like(x)
    x2 = x + noise_std*torch.randn_like(x)
    if drop_prob > 0:
        m1 = (torch.rand_like(x1[..., :1]) > drop_prob).float()
        m2 = (torch.rand_like(x2[..., :1]) > drop_prob).float()
        x1 = x1 * m1
        x2 = x2 * m2
    return x1, x2

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
    ap.add_argument('--proj', type=int, default=128)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--T', type=int, default=64)
    ap.add_argument('--D', type=int, default=16)
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--val_split', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save', type=str, default='seqclr_gru_best.pt')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    tr, va, _, _ = make_forda_sequence_loaders(batch_size=args.batch, seed=args.seed, return_index=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SeqCLRGRU(1, args.hidden, args.proj, args.layers).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    es = EarlyStopper(args.patience, 1e-4)

    best = None

    # Init Logger

    logger = ContinuousLogger(Path('results_run_rnn_seqclr'), 'run_rnn_seqclr', 'train')

    for epoch in range(1, args.epochs+1):
        net.train(); tr_loss = 0.0
        for x in tr:
            x = x.transpose(1, 2).to(device)
            x1,x2 = augment(x)
            z1 = net(x1); z2 = net(x2)
            loss = SeqCLRGRU.nt_xent(z1,z2,temperature=0.2)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(tr.dataset)

        net.eval(); va_loss = 0.0
        with torch.no_grad():
            for x in va:
                x = x.transpose(1, 2).to(device)
                x1,x2 = augment(x)
                z1=net(x1); z2=net(x2)
                loss = SeqCLRGRU.nt_xent(z1,z2,temperature=0.2)
                va_loss += loss.item() * x.size(0)
        va_loss /= len(va.dataset)

        improved = es.step(va_loss)
        if improved:
            best = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
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
