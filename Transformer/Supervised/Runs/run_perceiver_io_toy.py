import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_perceiver_io import PerceiverIO

class ToyMultiModal(Dataset):
    """Toy multimodal tokens: concatenate image patches (flattened small 8x8 RGB) + tabular tokens."""
    def __init__(self, n=4000, img_tokens=16, tab_tokens=8, token_dim=64, num_classes=4):
        self.X=[]; self.y=[]
        g = torch.Generator().manual_seed(0)
        for i in range(n):
            c = torch.randint(0,num_classes,(1,),generator=g).item()
            # generate token sequences with class-dependent mean
            img = torch.randn(img_tokens, token_dim, generator=g) + c*0.5
            tab = torch.randn(tab_tokens, token_dim, generator=g) + c*0.2
            tokens = torch.cat([img, tab], dim=0)
            self.X.append(tokens); self.y.append(c)
        self.X = torch.stack(self.X); self.y = torch.tensor(self.y)
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i): return self.X[i], self.y[i]


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
    ap = argparse.ArgumentParser(); ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    train_ds = ToyMultiModal(3200); val_ds = ToyMultiModal(800)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PerceiverIO(input_dim=64, latent_dim=256, latent_len=64, depth=6, heads=4, num_classes=4).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()

    best, bad, patience = 0.0, 0, 4

    # Init Logger

    logger = ContinuousLogger(Path('results_run_perceiver_io_toy'), 'run_perceiver_io_toy', 'train')

    for epoch in range(1, args.epochs+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(args.device), y.to(args.device)
            opt.zero_grad(); logits = model(x); loss = crit(logits,y); loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        acc = evaluate(model, val_loader, args.device)
        # Log

        msg = f"Epoch {epoch}: val_acc={acc:.4f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if acc > best + 1e-6:
            best=acc; bad=0; torch.save({'model': model.state_dict()}, 'PerceiverIO_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best val acc:', best)

if __name__ == '__main__':
    main()