import argparse
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_retnet import RetNetClassifier

class TSVText(Dataset):
    def __init__(self, n=4000, max_len=128):
        self.samples=[]; g=torch.Generator().manual_seed(0)
        for i in range(n):
            y = torch.randint(0,2,(1,),generator=g).item()
            tok = 3 if y==0 else 4
            ids = [2] + [tok]* (max_len-1)
            self.samples.append((y, torch.tensor(ids)))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): y,ids=self.samples[i]; return ids, y

def evaluate(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for ids,y in loader:
            ids,y = ids.to(device), y.to(device)
            logits = model(ids); pred = logits.argmax(-1)
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/max(total,1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    train = DataLoader(TSVText(3200), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(TSVText(800), batch_size=args.batch_size, shuffle=False)

    vocab = 8
    model = RetNetClassifier(vocab=vocab, num_classes=2).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()

    best, bad, patience = 0.0, 0, 3

    # Init Logger

    logger = ContinuousLogger(Path('results_run_retnet_text'), 'run_retnet_text', 'train')

    for epoch in range(1, args.epochs+1):
        model.train()
        for ids,y in train:
            ids,y = ids.to(args.device), y.to(args.device)
            opt.zero_grad(); logits=model(ids); loss=crit(logits,y); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
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
            best=acc; bad=0; torch.save({'model': model.state_dict()}, 'RetNet_best.pth')
        else:
            bad+=1
            if bad>=patience:
                print('Early stopping.'); break
    print('Done. Best val acc:', best)

if __name__ == '__main__':
    main()