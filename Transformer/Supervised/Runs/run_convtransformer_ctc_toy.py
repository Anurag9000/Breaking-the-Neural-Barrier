import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_convtransformer_ctc import ConvTransformerCTC

# Toy ASR-like dataset: random spectrograms and label sequences with simple pattern
class ToyASR(Dataset):
    def __init__(self, n=2000, T=128, F=64, vocab=20, max_lab=20):
        self.X=[]; self.Y=[]; self.Ly=[]; self.T=T
        g=torch.Generator().manual_seed(0)
        for i in range(n):
            x = torch.randn(1,T,F, generator=g)
            L = torch.randint(5, max_lab, (1,), generator=g).item()
            y = torch.randint(1, vocab, (L,), generator=g)  # 0 reserved for blank in CTC criterion call
            self.X.append(x); self.Y.append(y); self.Ly.append(L)
        self.X=torch.stack(self.X)
    def __len__(self): return self.X.size(0)
    def __getitem__(self,i): return self.X[i], self.Y[i], self.Ly[i]


def ctc_collate(batch):
    xs, ys, ls = zip(*batch)
    xs = torch.stack(xs)
    ycat = torch.cat(ys)
    ylen = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return xs, ycat, ylen


def evaluate(model, loader, device):
    model.eval(); tot=cnt=0
    with torch.no_grad():
        for x, ycat, ylen in loader:
            x = x.to(device)
            logits = model(x)  # B,T,V
            Tt = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
            logp = logits.log_softmax(-1).transpose(0,1)  # T,B,V for CTCLoss
            loss = nn.CTCLoss(blank=0, zero_infinity=True)(logp, ycat.to(device), Tt, ylen.to(device))
            tot += loss.item()*x.size(0); cnt += x.size(0)
    return tot/max(cnt,1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    train = DataLoader(ToyASR(1600), batch_size=args.batch_size, shuffle=True, collate_fn=ctc_collate)
    val = DataLoader(ToyASR(400), batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate)

    model = ConvTransformerCTC(vocab=21).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)

    best, bad, patience = 1e9, 0, 4

    # Init Logger

    logger = ContinuousLogger(Path('results_run_convtransformer_ctc_toy'), 'run_convtransformer_ctc_toy', 'train')

    for epoch in range(1, args.epochs+1):
        model.train()
        for x, ycat, ylen in train:
            x = x.to(args.device)
            Tt = torch.full((x.size(0),), model.forward(x).size(1), dtype=torch.long)
            logits = model(x)
            logp = logits.log_softmax(-1).transpose(0,1)
            opt.zero_grad(); loss = crit(logp, ycat.to(args.device), Tt, ylen.to(args.device)); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        val = evaluate(model, val, args.device)
        # Log

        msg = f"Epoch {epoch}: val_CTC={val:.4f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if val + 1e-6 < best:
            best=val; bad=0; torch.save({'model': model.state_dict()}, 'ConvTransformerCTC_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best val CTC:', best)

if __name__ == '__main__':
    main()