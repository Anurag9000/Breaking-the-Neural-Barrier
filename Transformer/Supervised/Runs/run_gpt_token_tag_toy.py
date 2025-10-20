import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_gpt_token_tag import GPTTagger

class ToyTag(Dataset):
    def __init__(self, n=4000, L=64):
        self.X=[]; self.Y=[]; g=torch.Generator().manual_seed(0)
        for i in range(n):
            seq = torch.randint(5,50,(L,),generator=g)
            y = (seq%2).long()  # tag parity
            x = torch.cat([torch.tensor([2]), seq])[:L]  # 2 as BOS
            self.X.append(x); self.Y.append(y)
        self.X=torch.stack(self.X); self.Y=torch.stack(self.Y)
    def __len__(self): return self.X.size(0)
    def __getitem__(self,i): return self.X[i], self.Y[i]


def evaluate(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for ids,tags in loader:
            ids,tags = ids.to(device), tags.to(device)
            logits = model(ids)
            pred = logits.argmax(-1)
            correct += (pred==tags).sum().item(); total += tags.numel()
    return correct/max(total,1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    train = DataLoader(ToyTag(3200), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(ToyTag(800), batch_size=args.batch_size, shuffle=False)

    model = GPTTagger(vocab=60, num_tags=2).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()

    best, bad, patience = 0.0, 0, 4
    for epoch in range(1, args.epochs+1):
        model.train()
        for ids,tags in train:
            ids,tags = ids.to(args.device), tags.to(args.device)
            opt.zero_grad(); logits=model(ids); loss=crit(logits.view(-1, logits.size(-1)), tags.view(-1)); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        acc = evaluate(model, val, args.device)
        print(f"Epoch {epoch}: token_acc={acc:.4f}")
        if acc > best + 1e-6:
            best=acc; bad=0; torch.save({'model': model.state_dict()}, 'GPTTagger_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best token acc:', best)

if __name__ == '__main__':
    main()
