import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_patchtst import PatchTST

class ToySeries(Dataset):
    def __init__(self, n=4000, T=256, C=4, num_classes=3):
        self.X=[]; self.y=[]
        g = torch.Generator().manual_seed(0)
        for i in range(n):
            cls = torch.randint(0,num_classes,(1,),generator=g).item()
            base = torch.linspace(0,1,T)
            series = []
            for c in range(C):
                if cls==0:
                    s = torch.sin(2*3.1415*(c+1)*base) + 0.05*torch.randn(T, generator=g)
                elif cls==1:
                    s = torch.sign(torch.sin(2*3.1415*(c+1)*base*0.5)) + 0.05*torch.randn(T, generator=g)
                else:
                    s = torch.randn(T, generator=g)
                series.append(s)
            x = torch.stack(series, dim=-1)
            self.X.append(x); self.y.append(cls)
        self.X = torch.stack(self.X); self.y = torch.tensor(self.y)
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


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
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    train_ds = ToySeries(3200)
    val_ds = ToySeries(800)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PatchTST(num_series=4, num_classes=3).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best, bad, patience = 0.0, 0, 4
    for epoch in range(1, args.epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(); logits = model(x); loss = crit(logits, y); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        acc = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch}: val_acc={acc:.4f}")
        if acc > best + 1e-6:
            best = acc; bad = 0; torch.save({'model': model.state_dict()}, 'PatchTST_best.pth')
        else:
            bad += 1
            if bad >= patience:
                print('Early stopping.'); break
    print('Done. Best val acc:', best)

if __name__ == '__main__':
    main()
