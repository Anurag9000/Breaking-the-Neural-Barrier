import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils as tvutils

from score_sde_model import ScoreUNet, ScoreSDE, VPSDE, count_parameters


def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def build_loaders(data_root, batch_size, val_split=5000, workers=4):
    mean=(0.5,0.5,0.5); std=(0.5,0.5,0.5)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(), transforms.Normalize(mean,std)
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    full_eval = datasets.CIFAR10(root=data_root, train=True, download=True, transform=eval_tf)
    n_total=len(full); n_val=val_split; n_train=n_total-n_val
    g=torch.Generator().manual_seed(123)
    train_subset,_ = random_split(full,[n_train,n_val],generator=g)
    _,val_subset = random_split(full_eval,[n_train,n_val],generator=g)
    tr = DataLoader(train_subset,batch_size=batch_size,shuffle=True,num_workers=workers,pin_memory=True)
    va = DataLoader(val_subset,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True)
    return tr, va


def train_epoch(model: ScoreSDE, loader, opt, device):
    model.train(); total=0.0
    for x,_ in loader:
        x=x.to(device)
        loss=model.loss(x)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def evaluate(model: ScoreSDE, loader, device):
    model.eval(); total=0.0
    for x,_ in loader:
        x=x.to(device); loss=model.loss(x); total+=loss.item()*x.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def save_samples(model: ScoreSDE, out_dir: Path, device, n=64):
    out_dir.mkdir(parents=True, exist_ok=True)
    x = model.pc_sample((n,3,32,32), device=device, steps=60)
    grid=(x+1)/2
    tvutils.save_image(grid, out_dir/"samples.png", nrow=8)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, default='./data')
    ap.add_argument('--save-dir', type=str, default='./artifacts_score_sde')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--val-split', type=int, default=5000)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    set_seed(42)
    tr,va = build_loaders(args.data_root, args.batch_size, args.val_split)

    device=torch.device(args.device)
    net=ScoreUNet(in_ch=3, base=64, ch_mult=(1,2,4), tdim=256, out_ch=3)
    sde=VPSDE(T=1.0, beta_min=0.1, beta_max=20.0)
    model=ScoreSDE(net, sde).to(device)

    print(f"Parameters: {count_parameters(model)/1e6:.2f} M")

    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best=float('inf'); bad=0; patience=30
    save_dir=Path(args.save_dir); (save_dir/"ckpts").mkdir(parents=True, exist_ok=True)
    best_path=save_dir/"ckpts"/"best.pt"

    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, tr, opt, device)
        va_loss = evaluate(model, va, device)
        print(f"Epoch {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
        if va_loss+1e-12 < best:
            best=va_loss; bad=0; torch.save({'model':model.state_dict()}, best_path)
        else:
            bad+=1
        if ep % 20 == 0:
            save_samples(model, save_dir/"samples", device)
        if bad>=patience:
            print('Early stopping.'); break

    if best_path.exists():
        state=torch.load(best_path,map_location=device)
        model.load_state_dict(state['model'])
    save_samples(model, save_dir/"samples", device)

if __name__=='__main__':
    main()
