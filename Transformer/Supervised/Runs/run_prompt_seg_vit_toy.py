import argparse, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_prompt_seg_vit import PromptableSegViT

class ShapesPromptSeg(Dataset):
    def __init__(self, n=1000, size=128):
        self.n=n; self.size=size; random.seed(0)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        S=self.size
        img = torch.zeros(3,S,S)
        mask = torch.zeros(S,S, dtype=torch.long)
        # rectangle
        x0=random.randint(0,S-40); y0=random.randint(0,S-40)
        w=random.randint(20,40); h=random.randint(20,40)
        x1=min(S-1,x0+w); y1=min(S-1,y0+h)
        img[:, y0:y1, x0:x1] = torch.rand(3,1,1)
        mask[y0:y1, x0:x1] = 1
        # pick a prompt point inside the rectangle
        px = random.randint(x0, x1-1); py = random.randint(y0, y1-1)
        point = torch.tensor([px/S, py/S], dtype=torch.float32)
        return img, mask, point


def evaluate(model, loader, device):
    model.eval(); miou=0.0; n=0
    with torch.no_grad():
        for x,y,p in loader:
            x,y,p = x.to(device), y.to(device), p.to(device)
            logits = model(x,p); pred = (logits>0).long()
            inter = ((pred==1) & (y==1)).sum().item(); union = ((pred==1) | (y==1)).sum().item()
            iou = inter/max(union,1)
            miou += iou; n+=1
    return miou/max(n,1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    train_ds = ShapesPromptSeg(800); val_ds = ShapesPromptSeg(200)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PromptableSegViT().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); crit = nn.BCEWithLogitsLoss()

    best, bad, patience = 0.0, 0, 6
    for epoch in range(1, args.epochs+1):
        model.train()
        for x,y,p in train_loader:
            x,y,p = x.to(args.device), y.to(args.device), p.to(args.device)
            opt.zero_grad(); logits = model(x,p); loss = crit(logits, (y==1).float()); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        score = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch}: val_IoU={score:.4f}")
        if score > best + 1e-6:
            best=score; bad=0; torch.save({'model': model.state_dict()}, 'PromptSegViT_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best val IoU:', best)

if __name__ == '__main__':
    main()
