import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_graphormer_toy import GraphormerToy

class ToyGraphs(Dataset):
    def __init__(self, n=2000, N=16, F=8, num_classes=2):
        self.X=[]; self.SPD=[]; self.y=[]
        g = torch.Generator().manual_seed(0)
        for i in range(n):
            cls = torch.randint(0,num_classes,(1,),generator=g).item()
            # random adjacency with different densities per class
            p = 0.2 if cls==0 else 0.5
            A = (torch.rand(N,N,generator=g) < p).float(); A = torch.triu(A,1); A = A + A.t(); A.fill_diagonal_(0)
            # shortest path distances via Floyd-Warshall (toy, N small)
            dist = torch.full((N,N), 1e6)
            dist.fill_diagonal_(0)
            edges = (A>0).nonzero()
            for u,v in edges:
                dist[u,v]=1; dist[v,u]=1
            for k in range(N):
                for i2 in range(N):
                    for j2 in range(N):
                        dist[i2,j2] = torch.minimum(dist[i2,j2], dist[i2,k]+dist[k,j2])
            dist[dist==1e6]=N
            x = torch.randn(N,F,generator=g)
            self.X.append(x); self.SPD.append(dist); self.y.append(cls)
        self.X = torch.stack(self.X); self.SPD = torch.stack(self.SPD); self.y = torch.tensor(self.y)
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx): return self.X[idx], self.SPD[idx], self.y[idx]


def evaluate(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for x, spd, y in loader:
            x, spd, y = x.to(device), spd.to(device), y.to(device)
            logits = model(x, spd)
            pred = logits.argmax(-1)
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/max(total,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    train_ds = ToyGraphs(1600)
    val_ds = ToyGraphs(400)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GraphormerToy(num_node_feats=8, num_classes=2).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best, bad, patience = 0.0, 0, 5
    for epoch in range(1, args.epochs+1):
        model.train()
        for x, spd, y in train_loader:
            x, spd, y = x.to(args.device), spd.to(args.device), y.to(args.device)
            opt.zero_grad(); logits = model(x, spd); loss = crit(logits, y); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        acc = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch}: val_acc={acc:.4f}")
        if acc > best + 1e-6:
            best = acc; bad = 0; torch.save({'model': model.state_dict()}, 'GraphormerToy_best.pth')
        else:
            bad += 1
            if bad >= patience:
                print('Early stopping.'); break
    print('Done. Best val acc:', best)

if __name__ == '__main__':
    main()
