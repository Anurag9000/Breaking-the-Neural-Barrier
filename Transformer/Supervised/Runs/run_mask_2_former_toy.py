import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_mask2former_toy import Mask2FormerToy

class ShapesSeg(Dataset):
    def __init__(self, n=1000, size=128, num_classes=3):
        self.n=n; self.size=size; self.num_classes=num_classes
        random.seed(1)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        S=self.size
        img = torch.zeros(3,S,S)
        mask = torch.zeros(S,S, dtype=torch.long)
        for cls in [1,2]:
            x0=random.randint(0,S-40); y0=random.randint(0,S-40)
            w=random.randint(20,40); h=random.randint(20,40)
            x1=min(S-1,x0+w); y1=min(S-1,y0+h)
            img[:, y0:y1, x0:x1] = torch.rand(3,1,1)
            mask[y0:y1, x0:x1] = cls
        return img, mask


def soft_iou_loss(pred_mask_logits, target_mask, eps=1e-6):
    # pred: (B,Q,H,W) raw logits -> pick max over queries per class outside for simplicity; here supervise union of predicted masks per class via BCE
    B,Q,H,W = pred_mask_logits.shape
    target_onehot = torch.nn.functional.one_hot(target_mask, num_classes=3).permute(0,3,1,2).float()  # B, K, H, W
    pred_sig = pred_mask_logits.sigmoid()  # B,Q,H,W
    # merge queries by max
    pred_merged, _ = pred_sig.max(dim=1, keepdim=True)  # B,1,H,W (treat as foreground vs background)
    target_fg = (target_mask>0).float().unsqueeze(1)
    inter = (pred_merged*target_fg).sum(dim=(2,3))
    union = (pred_merged + target_fg - pred_merged*target_fg).sum(dim=(2,3))
    iou = (inter+eps)/(union+eps)
    return 1 - iou.mean()


def evaluate(model, loader, device):
    model.eval(); miou=0.0; n=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            cls, masks = model(x)
            pred = (masks.sigmoid().max(1).values>0.5).long()  # foreground mask only
            # crude IoU on foreground
            inter = ((pred==1) & (y>0)).sum().item(); union = ((pred==1) | (y>0)).sum().item()
            iou = inter/max(union,1)
            miou += iou; n+=1
    return miou/max(n,1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    train_ds = ShapesSeg(800); val_ds = ShapesSeg(200)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = Mask2FormerToy(num_classes=3).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best, bad, patience = 0.0, 0, 8
    for epoch in range(1, args.epochs+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(args.device), y.to(args.device)
            opt.zero_grad(); cls, masks = model(x)
            # simple loss: BCE IoU-like on foreground + CE on classes (assign all queries to foreground class)
            loss = soft_iou_loss(masks, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        score = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch}: val_fgIoU={score:.4f}")
        if score > best + 1e-6:
            best=score; bad=0; torch.save({'model': model.state_dict()}, 'Mask2FormerToy_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best val IoU:', best)

if __name__ == '__main__':
    main()
