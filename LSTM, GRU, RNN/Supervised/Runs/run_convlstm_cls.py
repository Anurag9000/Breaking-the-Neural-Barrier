import argparse, os, random
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
from torch.utils.data import Dataset, DataLoader

from convlstm_cls import ConvLSTMClassifier, ConvLSTMConfig


class SyntheticVideoCls(Dataset):
    """Random moving-blob videos with class label equal to blob quadrant at last frame.
    Small toy to exercise ConvLSTM in supervised many-to-one classification.
    """
    def __init__(self, n:int, T_min:int, T_max:int, H:int, W:int, seed:int=777):
        rng = random.Random(seed)
        self.samples = []
        for _ in range(n):
            T = rng.randint(T_min, T_max)
            frames = []
            x = rng.randint(0, W-1); y = rng.randint(0, H-1)
            vx = rng.choice([-1,1]); vy = rng.choice([-1,1])
            for t in range(T):
                img = [[0.0]*W for _ in range(H)]
                img[y][x] = 1.0
                frames.append(img)
                x = max(0, min(W-1, x+vx)); y = max(0, min(H-1, y+vy))
            # class: quadrant of final position
            q = (y >= H//2)*2 + (x >= W//2)
            self.samples.append((frames, T, q))
        self.H, self.W = H, W

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        frames, T, q = self.samples[i]
        # to tensor (T,1,H,W)
        import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
        vid = torch.tensor(frames, dtype=torch.float32).unsqueeze(1)
        return vid, T, q


def collate_video(batch):
    B = len(batch)
    T_max = max(T for _, T, _ in batch)
    H = batch[0][0].size(-2); W = batch[0][0].size(-1)
    vids = torch.zeros(B, T_max, 1, H, W)
    lengths = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)
    for i, (vid, T, q) in enumerate(batch):
        vids[i, :T] = vid
        lengths[i] = T
        labels[i] = q
    return vids, lengths, labels


class EarlyStopper:
    def __init__(self, patience):
        self.p=patience; self.best=float('inf'); self.bad=0; self.state=None
    def step(self, v, m):
        if v < self.best - 1e-7:
            self.best=v; self.bad=0; self.state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        else: self.bad+=1
        return self.bad>=self.p
    def restore(self, m):
        if self.state is not None: m.load_state_dict(self.state)


def train_epoch(model, loader, opt, crit, device):
    model.train(); tot=0.0; n=0
    for vids, lengths, labels in loader:
        vids, lengths, labels = vids.to(device), lengths.to(device), labels.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(vids, lengths)
        loss = crit(logits, labels)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += float(loss.item()) * labels.size(0); n += labels.size(0)
    return tot/max(1,n)


def eval_epoch(model, loader, crit, device):
    model.eval(); tot=0.0; n=0; corr=0
    with torch.no_grad():
        for vids, lengths, labels in loader:
            vids, lengths, labels = vids.to(device), lengths.to(device), labels.to(device)
            logits = model(vids, lengths)
            loss = crit(logits, labels)
            tot += float(loss.item()) * labels.size(0); n += labels.size(0)
            corr += int((logits.argmax(-1)==labels).sum().item())
    return tot/max(1,n), corr/max(1,n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-n', type=int, default=4000)
    ap.add_argument('--val-n', type=int, default=500)
    ap.add_argument('--test-n', type=int, default=500)
    ap.add_argument('--t-min', type=int, default=8)
    ap.add_argument('--t-max', type=int, default=16)
    ap.add_argument('--H', type=int, default=16)
    ap.add_argument('--W', type=int, default=16)

    ap.add_argument('--in-channels', type=int, default=1)
    ap.add_argument('--hidden-channels', type=int, default=32)
    ap.add_argument('--kernel-size', type=int, default=3)
    ap.add_argument('--num-layers', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--num-classes', type=int, default=4)

    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--outdir', type=str, default='results_lstm')
    args = ap.parse_args()

    random.seed(777); torch.manual_seed(777)
    os.makedirs(args.outdir, exist_ok=True)

    train_ds = SyntheticVideoCls(args.train_n, args.t_min, args.t_max, args.H, args.W, seed=101)
    val_ds = SyntheticVideoCls(args.val_n, args.t_min, args.t_max, args.H, args.W, seed=102)
    test_ds = SyntheticVideoCls(args.test_n, args.t_min, args.t_max, args.H, args.W, seed=103)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_video)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_video)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_video)

    cfg = ConvLSTMConfig(in_channels=args.in_channels, hidden_channels=args.hidden_channels,
                         kernel_size=args.kernel_size, num_layers=args.num_layers,
                         dropout=args.dropout, num_classes=args.num_classes)
    model = ConvLSTMClassifier(cfg).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    es = EarlyStopper(args.patience)


    # Init Logger


    logger = ContinuousLogger(Path('results_run_convlstm_cls'), 'run_convlstm_cls', 'train')


    for epoch in range(1, args.max_epochs+1):
        tr = train_epoch(model, train_loader, opt, crit, args.device)
        vl, vacc = eval_epoch(model, val_loader, crit, args.device)
        stop = es.step(vl, model)
        # Log

        msg = f'Epoch {epoch:03d} | train_loss={tr:.4f} | val_loss={vl:.4f} | val_acc={vacc:.4f}'

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if stop:
            print('Early stopping.'); break

    es.restore(model)
    tl, ta = eval_epoch(model, test_loader, crit, args.device)
    print(f'TEST | loss={tl:.4f} | acc={ta:.4f}')

    torch.save({'model_state': model.state_dict(), 'config': cfg.__dict__, 'test_loss': tl, 'test_acc': ta},
               os.path.join(args.outdir, 'convlstm_cls.pt'))

if __name__ == '__main__':
    main()