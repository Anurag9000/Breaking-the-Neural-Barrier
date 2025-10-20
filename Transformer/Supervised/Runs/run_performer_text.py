import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_performer import PerformerEncoder

class TSVText(Dataset):
    def __init__(self, n=4000, max_len=256):
        self.samples=[]; g=torch.Generator().manual_seed(3)
        for i in range(n):
            y = torch.randint(0,2,(1,),generator=g).item()
            tok = 3 if y==0 else 4
            ids = torch.full((max_len,), tok); ids[0]=2
            self.samples.append((ids, y))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i]


def evaluate(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for ids,y in loader:
            ids,y = ids.to(device), y.to(device)
            logits = model(ids); pred = logits.argmax(-1)
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/max(total,1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    train = DataLoader(TSVText(3200), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(TSVText(800), batch_size=args.batch_size, shuffle=False)

    model = PerformerEncoder(vocab=16, num_classes=2, dim=256, depth=6, heads=8).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()

    best, bad, patience = 0.0, 0, 4
    for epoch in range(1, args.epochs+1):
        model.train()
        for ids,y in train:
            ids,y = ids.to(args.device), y.to(args.device)
            opt.zero_grad(); logits=model(ids); loss=crit(logits,y); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        acc = evaluate(model, val, args.device)
        print(f"Epoch {epoch}: val_acc={acc:.4f}")
        if acc > best + 1e-6:
            best=acc; bad=0; torch.save({'model': model.state_dict()}, 'Performer_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best val acc:', best)

if __name__ == '__main__':
    main()
