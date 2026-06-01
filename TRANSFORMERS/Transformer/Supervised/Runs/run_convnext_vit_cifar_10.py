import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from utils.adp_logging import ContinuousLogger.nn as nn
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from utils.adp_logging import ContinuousLoggervision as tv
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from utils.adp_logging import ContinuousLoggervision.transforms as T
from torch.utils.data import DataLoader, random_split

from model_convnext_vit import HybridConvNeXtViT


def evaluate(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x); pred = logits.argmax(-1)
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/max(total,1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=6); ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    tf_eval = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])

    train_full = tv.datasets.CIFAR10('./data', train=True, download=True, transform=tf_train)
    val_len = 5000; train_len = len(train_full)-val_len
    train_ds, _ = random_split(train_full, [train_len, val_len])
    val_ds = tv.datasets.CIFAR10('./data', train=True, download=True, transform=tf_eval)
    val_ds, _ = random_split(val_ds, [val_len, len(val_ds)-val_len])
    test_ds = tv.datasets.CIFAR10('./data', train=False, download=True, transform=tf_eval)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = HybridConvNeXtViT(num_classes=10).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()

    best, bad=0.0,0

    # Init Logger

    logger = ContinuousLogger(Path('results_run_convnext_vit_cifar_10'), 'run_convnext_vit_cifar_10', 'train')

    for epoch in range(1, args.epochs+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(args.device), y.to(args.device)
            opt.zero_grad(); logits=model(x); loss=crit(logits,y); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
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
            best=acc; bad=0; torch.save({'model': model.state_dict()}, 'ConvNeXtViT_CIFAR10_best.pth')
        else:
            bad+=1
            if bad>=args.patience:
                print('Early stopping.'); break
    test = evaluate(model, test_loader, args.device)
    print('Done. Best val acc:', best, ' Test acc:', test)

if __name__ == '__main__':
    main()