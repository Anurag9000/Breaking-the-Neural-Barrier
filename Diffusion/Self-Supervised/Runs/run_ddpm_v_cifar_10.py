import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
from torchvision import utils as tvutils

from ddpm_v_model import UNet, DDPMv, DiffCfg, count_parameters
from runs._common_real_image import make_real_image_loaders


def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def train_epoch(model: DDPMv, loader, opt, device):
    model.train(); tot=0.0
    for x,_ in loader:
        x=x.to(device)
        t=torch.randint(0, model.cfg.timesteps, (x.size(0),), device=device)
        loss=model.p_losses(x,t)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tot+=loss.item()*x.size(0)
    return tot/len(loader.dataset)

@torch.no_grad()
def evaluate(model: DDPMv, loader, device):
    model.eval(); tot=0.0
    for x,_ in loader:
        x=x.to(device); t=torch.randint(0, model.cfg.timesteps,(x.size(0),),device=device)
        loss=model.p_losses(x,t); tot+=loss.item()*x.size(0)
    return tot/len(loader.dataset)

@torch.no_grad()
def save_samples(model: DDPMv, out_dir: Path, device, n=64):
    out_dir.mkdir(parents=True, exist_ok=True)
    x=model.sample((n,3,32,32), device=device)
    grid=(x+1)/2
    tvutils.save_image(grid, out_dir/"samples.png", nrow=8)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, default='./data')
    ap.add_argument('--save-dir', type=str, default='./artifacts_ddpm_v')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--val-split', type=int, default=5000)
    ap.add_argument('--timesteps', type=int, default=1000)
    ap.add_argument('--beta-start', type=float, default=1e-4)
    ap.add_argument('--beta-end', type=float, default=2e-2)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    set_seed(42)
    tr,va,_=make_real_image_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_ratio=0.1,
        test_ratio=0.1,
        num_workers=0,
        image_size=32,
    )
    device=torch.device(args.device)

    net=UNet(in_ch=3, base=64, ch_mult=(1,2,4), tdim=256, out_ch=3)
    model=DDPMv(net, DiffCfg(args.timesteps, args.beta_start, args.beta_end)).to(device)
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
        if bad>=30:
            print('Early stopping.'); break

    if best_path.exists():
        state=torch.load(best_path,map_location=device); model.load_state_dict(state['model'])
    save_samples(model, save_dir/"samples", device)

if __name__=='__main__':
    main()
