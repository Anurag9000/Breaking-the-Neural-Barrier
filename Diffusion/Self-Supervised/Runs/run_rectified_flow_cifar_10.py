import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
from torchvision import utils as tvutils

from rectified_flow_model import RFUNet, RectifiedFlow, count_parameters
from runs._common_real_image import make_real_image_loaders


def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def train_epoch(model: RectifiedFlow, loader, opt, device):
    model.train(); tot=0.0
    for x,_ in loader:
        x=x.to(device)
        loss=model.loss(x)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tot+=loss.item()*x.size(0)
    return tot/len(loader.dataset)

@torch.no_grad()
def evaluate(model: RectifiedFlow, loader, device):
    model.eval(); tot=0.0
    for x,_ in loader:
        x=x.to(device); loss=model.loss(x); tot+=loss.item()*x.size(0)
    return tot/len(loader.dataset)

@torch.no_grad()
def save_samples(model: RectifiedFlow, out_dir: Path, device, n=64):
    out_dir.mkdir(parents=True, exist_ok=True)
    x=model.sample((n,3,32,32), device=device, steps=60)
    grid=(x+1)/2
    tvutils.save_image(grid, out_dir/"samples.png", nrow=8)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, default='./data')
    ap.add_argument('--save-dir', type=str, default='./artifacts_rectified_flow')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--val-split', type=int, default=5000)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    set_seed(42)
    tr,va,_=make_real_image_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_ratio=0.1,
        test_ratio=0.1,
        num_workers=4,
        image_size=32,
    )
    device=torch.device(args.device)

    net=RFUNet(in_ch=3, base=64, ch_mult=(1,2,4), tdim=256, out_ch=3)
    model=RectifiedFlow(net).to(device)
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
        if ep%20==0: save_samples(model, save_dir/"samples", device)
        if bad>=patience:
            print('Early stopping.'); break

    if best_path.exists():
        state=torch.load(best_path,map_location=device); model.load_state_dict(state['model'])
    save_samples(model, save_dir/"samples", device)

if __name__=='__main__':
    main()
