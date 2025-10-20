import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_conformer import ConformerEncoder

# Simple synthetic audio features dataset: classify sine vs square vs noise (toy)
class ToyAudio(Dataset):
    def __init__(self, n=3000, T=400, F=80, num_classes=3):
        super().__init__(); self.n=n; self.T=T; self.F=F; self.num_classes=num_classes
        g = torch.Generator().manual_seed(0)
        self.data=[]
        for i in range(n):
            c = torch.randint(0,num_classes,(1,),generator=g).item()
            t = torch.linspace(0,1,T)
            if c==0:
                s = torch.sin(2*math.pi*5*t)
            elif c==1:
                s = torch.sign(torch.sin(2*math.pi*3*t))
            else:
                s = torch.randn(T)
            # expand to F features by stacking shifted copies
            feats = torch.stack([torch.roll(s, shifts=j) for j in range(F)], dim=-1).float()
            self.data.append((feats, c))
    def __len__(self): return self.n
    def __getitem__(self, idx): return self.data[idx]


def evaluate(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for feats, y in loader:
            feats, y = feats.to(device), y.to(device)
            logits = model(feats)
            pred = logits.argmax(-1)
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/max(total,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    train_ds = ToyAudio(2400)
    val_ds = ToyAudio(600)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ConformerEncoder(in_feats=80, num_classes=3).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best, bad, patience = 0.0, 0, 4
    for epoch in range(1, args.epochs+1):
        model.train()
        for feats, y in train_loader:
            feats, y = feats.to(args.device), y.to(args.device)
            opt.zero_grad(); logits = model(feats); loss = crit(logits, y); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        acc = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch}: val_acc={acc:.4f}")
        if acc > best + 1e-6:
            best = acc; bad = 0; torch.save({'model': model.state_dict()}, 'Conformer_best.pth')
        else:
            bad += 1
            if bad >= patience:
                print('Early stopping.'); break
    print('Done. Best val acc:', best)

if __name__ == '__main__':
    main()
