import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLoggervision as tv
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLoggervision.transforms as T
from torch.utils.data import DataLoader

from model_perceiver import Perceiver

class CIFARAsTokens(torch.utils.data.Dataset):
    def __init__(self, train=True):
        tf = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
        self.ds = tv.datasets.CIFAR10(root='./data', train=train, download=True, transform=tf)
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x,y = self.ds[i]
        # downsample to 8x8 patches to keep token count small
        x = torch.nn.functional.avg_pool2d(x, kernel_size=4)  # 8x8
        tokens = x.flatten(1).transpose(0,1)  # (64, 3)
        tokens = tokens.view(64, 3)
        # project to 64-dim token via padding/linear in the model; here just return 3-dim and model expects 64 -> we pad
        pad = torch.zeros(64, 61); tokens = torch.cat([tokens, pad], dim=1)
        return tokens, y


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
    ap = argparse.ArgumentParser(); ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    train = DataLoader(CIFARAsTokens(True), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val = DataLoader(CIFARAsTokens(False), batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = Perceiver(input_dim=64, latent_dim=256, latent_len=64, depth=6, heads=4, num_classes=10).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()

    best, bad, patience = 0.0, 0, 6

    # Init Logger

    logger = ContinuousLogger(Path('results_run_perceiver_cifar_10'), 'run_perceiver_cifar_10', 'train')

    for epoch in range(1, args.epochs+1):
        model.train()
        for x,y in train:
            x,y = x.to(args.device), y.to(args.device)
            opt.zero_grad(); logits=model(x); loss=crit(logits,y); loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        acc = evaluate(model, val, args.device)
        # Log

        msg = f"Epoch {epoch}: val_acc={acc:.4f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if acc > best + 1e-6:
            best=acc; bad=0; torch.save({'model': model.state_dict()}, 'Perceiver_CIFAR10_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best val acc:', best)

if __name__ == '__main__':
    main()