import argparse, random
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_transunet import TransUNet

class ShapesSeg(Dataset):
    def __init__(self, n=1000, size=128, num_classes=3):
        self.n=n; self.size=size; self.num_classes=num_classes
        random.seed(3)
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


def mIoU(pred, target, K):
    ious=[]
    for k in range(1,K):
        p=(pred==k); t=(target==k)
        inter=(p & t).sum().item(); union=(p|t).sum().item()
        if union==0: continue
        ious.append(inter/union)
    return sum(ious)/max(len(ious),1)


def evaluate(model, loader, device, K):
    model.eval(); miou=0.0; n=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            miou += mIoU(pred.cpu(), y.cpu(), K); n+=1
    return miou/max(n,1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    train_ds = ShapesSeg(800); val_ds = ShapesSeg(200)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TransUNet(num_classes=3).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()

    best, bad, patience = 0.0, 0, 6

    # Init Logger

    logger = ContinuousLogger(Path('results_run_transunet_toy'), 'run_transunet_toy', 'train')

    for epoch in range(1, args.epochs+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(args.device), y.to(args.device)
            opt.zero_grad(); logits=model(x); loss=crit(logits, y); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        miou = evaluate(model, val_loader, args.device, 3)
        # Log

        msg = f"Epoch {epoch}: val_mIoU={miou:.4f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if miou > best + 1e-6:
            best=miou; bad=0; torch.save({'model': model.state_dict()}, 'TransUNet_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best val mIoU:', best)

if __name__ == '__main__':
    main()