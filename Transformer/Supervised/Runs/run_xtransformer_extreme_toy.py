import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_xtransformer_extreme import XTransformer

class ToyExtreme(Dataset):
    def __init__(self, n=5000, L=64, num_labels=100):
        self.X=[]; self.Y=[]; g=torch.Generator().manual_seed(0)
        for i in range(n):
            ids = torch.randint(5,100,(L,),generator=g)
            # assign 3 random labels correlated with parity
            y = torch.zeros(num_labels)
            base = (ids.sum().item() % num_labels)
            idxs = [(base+j*7)%num_labels for j in range(3)]
            y[idxs]=1
            self.X.append(torch.cat([torch.tensor([2]), ids])[:L]); self.Y.append(y)
        self.X=torch.stack(self.X); self.Y=torch.stack(self.Y)
    def __len__(self): return self.X.size(0)
    def __getitem__(self,i): return self.X[i], self.Y[i]


def evaluate(model, loader, device):
    model.eval(); tot_prec=0.0; n=0
    with torch.no_grad():
        for ids,y in loader:
            ids,y = ids.to(device), y.to(device)
            logits = model(ids)
            topk = logits.topk(3, dim=-1).indices
            correct = (y.gather(1, topk)>0).float().mean().item()
            tot_prec += correct; n+=1
    return tot_prec/max(n,1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    LBL=100
    train = DataLoader(ToyExtreme(4000, num_labels=LBL), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(ToyExtreme(1000, num_labels=LBL), batch_size=args.batch_size, shuffle=False)

    model = XTransformer(vocab=120, num_labels=LBL).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()

    best, bad, patience = 0.0, 0, 4
    for epoch in range(1, args.epochs+1):
        model.train()
        for ids,y in train:
            ids,y = ids.to(args.device), y.to(args.device)
            opt.zero_grad(); logits=model(ids); loss=crit(logits,y); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        p3 = evaluate(model, val, args.device)
        print(f"Epoch {epoch}: P@3={p3:.4f}")
        if p3 > best + 1e-6:
            best=p3; bad=0; torch.save({'model': model.state_dict()}, 'XTransformerExtreme_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best P@3:', best)

if __name__ == '__main__':
    main()
