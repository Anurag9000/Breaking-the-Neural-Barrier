import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils as tvutils

from cold_diffusion_model import ColdUNet, ColdDiffusion, ColdCfg, count_parameters


def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def build_loaders(data_root, batch, val_split=5000, workers=4):
    mean=(0.5,0.5,0.5); std=(0.5,0.5,0.5)
    tr_tf=transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(), transforms.Normalize(mean,std)
    ])
    ev_tf=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    full=datasets.CIFAR10(root=data_root, train=True, download=True, transform=tr_tf)
    evalset=datasets.CIFAR10(root=data_root, train=True, download=True, transform=ev_tf)
    n=len(full); n_val=val_split; n_tr=n-n_val
    g=torch.Generator().manual_seed(123)
    tr,_=random_split(full,[n_tr,n_val],generator=g)
    _,va=random_split(evalset,[n_tr,n_val],generator=g)
    dl_tr=DataLoader(tr,batch_size=batch,shuffle=True,num_workers=workers,pin_memory=True)
    dl_va=DataLoader(va,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=True)
    return dl_tr, dl_va


def train_epoch(model: ColdDiffusion, loader, opt, device):
    model.train(); tot=0.0
    for x,_ in loader:
        x=x.to(device)
        loss=model.loss(x)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tot+=loss.item()*x.size(0)
    return tot/len(loader.dataset)

@torch.no_grad()
def evaluate(model: ColdDiffusion, loader, device):
    model.eval(); tot=0.0
    for x,_ in loader:
        x=x.to(device); loss=model.loss(x); tot+=loss.item()*x.size(0)
    return tot/len(loader.dataset)

@torch.no_grad()
def save_inversions(model: ColdDiffusion, out_dir: Path, device, n=16):
    out_dir.mkdir(parents=True, exist_ok=True)
    x=torch.randn(n,3,32,32, device=device)
    # create corrupted samples from random images (for demo)
    x=(x.clamp(-1,1))
    inv=model.inverse(x, steps=20)
    grid=(inv+1)/2
    tvutils.save_image(grid, out_dir/"inversions.png", nrow=4)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, default='./data')
    ap.add_argument('--save-dir', type=str, default='./artifacts_cold')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--val-split', type=int, default=5000)
    ap.add_argument('--mode', type=str, choices=['blur','mask'], default='blur')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    set_seed(42)
    tr,va=build_loaders(args.data_root, args.batch_size, args.val_split)
    device=torch.device(args.device)

    net=ColdUNet(in_ch=3, base=64, ch_mult=(1,2,4), tdim=256, out_ch=3)
    model=ColdDiffusion(net, ColdCfg(T=1000, mode=args.mode)).to(device)
    print(f"Parameters: {count_parameters(model)/1e6:.2f} M")

    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best=float('inf'); bad=0; patience=30
    save_dir=Path(args.save_dir); (save_dir/"ckpts").mkdir(parents=True, exist_ok=True)
    best_path=save_dir/"ckpts"/"best.pt"

    for ep in range(1, args.epochs+1):
        tr_loss=train_epoch(model,tr,opt,device)
        va_loss=evaluate(model,va,device)
        print(f"Epoch {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
        if va_loss+1e-12 < best:
            best=va_loss; bad=0; torch.save({'model':model.state_dict()}, best_path)
        else:
            bad+=1
        if ep%20==0: save_inversions(model, save_dir/"samples", device)
        if bad>=patience:
            print('Early stopping.'); break

    if best_path.exists():
        state=torch.load(best_path,map_location=device); model.load_state_dict(state['model'])
    save_inversions(model, save_dir/"samples", device)

if __name__=='__main__':
    main()
