import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_tabtransformer import TabTransformer

class ToyTabular(Dataset):
    def __init__(self, n=10000, fields=(50, 30, 20, 10), num_classes=3):
        self.X=[]; self.y=[]; self.fields=fields
        g = torch.Generator().manual_seed(0)
        for i in range(n):
            x = [torch.randint(0,f,(1,),generator=g).item() for f in fields]
            # class depends on parity and range thresholds (nonlinearity)
            s = sum((x[j] % (j+2)) for j in range(len(fields)))
            c = (s % num_classes)
            self.X.append(torch.tensor(x)); self.y.append(c)
        self.X = torch.stack(self.X); self.y = torch.tensor(self.y)
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i): return self.X[i], self.y[i]


def evaluate(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(-1)
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/max(total,1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    fields = (50,30,20,10)
    train_ds = ToyTabular(8000, fields); val_ds = ToyTabular(2000, fields)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TabTransformer(num_categories_per_field=fields, num_classes=3).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()

    best, bad, patience = 0.0, 0, 3

    # Init Logger

    logger = ContinuousLogger(Path('results_run_tabtransformer_toy'), 'run_tabtransformer_toy', 'train')

    for epoch in range(1, args.epochs+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(args.device), y.to(args.device)
            opt.zero_grad(); logits=model(x); loss=crit(logits,y); loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        acc = evaluate(model, val_loader, args.device)
        # Log

        msg = f"Epoch {epoch}: val_acc={acc:.4f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if acc > best + 1e-6:
            best=acc; bad=0; torch.save({'model': model.state_dict()}, 'TabTransformer_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best val acc:', best)

if __name__ == '__main__':
    main()