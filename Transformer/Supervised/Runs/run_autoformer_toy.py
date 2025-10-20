import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_autoformer import Autoformer

class ToyForecast(Dataset):
    def __init__(self, n=2000, in_len=96, pred_len=24):
        self.Xi=[]; self.Xo=[]; self.Y=[]; self.in_len=in_len; self.pred_len=pred_len
        g = torch.Generator().manual_seed(1)
        for i in range(n):
            f = torch.rand(1, generator=g)*2 + 0.5
            t = torch.linspace(0,10,in_len+pred_len)
            trend = 0.05*t
            y = trend + torch.sin(f*t) + 0.1*torch.randn_like(t, generator=g)
            self.Xi.append(y[:in_len].unsqueeze(-1))
            self.Xo.append(torch.zeros(pred_len,1))
            self.Y.append(y[in_len:].unsqueeze(-1))
        self.Xi=torch.stack(self.Xi); self.Xo=torch.stack(self.Xo); self.Y=torch.stack(self.Y)
    def __len__(self): return self.Xi.size(0)
    def __getitem__(self,i): return self.Xi[i], self.Xo[i], self.Y[i]


def evaluate(model, loader, device):
    model.eval(); se=cnt=0
    with torch.no_grad():
        for xi, xo, y in loader:
            xi,xo,y = xi.to(device), xo.to(device), y.to(device)
            pred = model(xi,xo)
            se += ((pred-y)**2).sum().item(); cnt += y.numel()
    return (se/cnt) ** 0.5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    train_ds = ToyForecast(1600); val_ds = ToyForecast(400)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = Autoformer().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); crit = nn.MSELoss()

    best, bad, patience = 1e9, 0, 4
    for epoch in range(1, args.epochs+1):
        model.train()
        for xi,xo,y in train_loader:
            xi,xo,y = xi.to(args.device), xo.to(args.device), y.to(args.device)
            opt.zero_grad(); pred = model(xi,xo); loss = crit(pred,y); loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        rmse = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch}: val_RMSE={rmse:.4f}")
        if rmse + 1e-6 < best:
            best=rmse; bad=0; torch.save({'model': model.state_dict()}, 'Autoformer_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best val RMSE:', best)

if __name__ == '__main__':
    main()
