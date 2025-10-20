import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as T

from adp_vq_vae import VQVAEConfig, VQVAE

MEAN, STD=(0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)

def loaders(root,batch):
    tr=T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN,STD)])
    te=T.Compose([T.ToTensor(), T.Normalize(MEAN,STD)])
    dtr=torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tr)
    dte=torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=te)
    return torch.utils.data.DataLoader(dtr,batch_size=batch,shuffle=True,num_workers=2,pin_memory=True), \
           torch.utils.data.DataLoader(dte,batch_size=batch,shuffle=False,num_workers=2,pin_memory=True)

def renorm(x,dev):
    return torch.clamp(x*torch.tensor(STD,device=dev).view(1,3,1,1)+torch.tensor(MEAN,device=dev).view(1,3,1,1),0,1)

def train(args):
    dev=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    tr,te=loaders(args.data,args.batch)
    cfg=VQVAEConfig(latent_dim=args.latent, n_codes=args.codes, width=args.width, depth=args.depth, beta_commit=args.beta_c)
    model=VQVAE(cfg).to(dev)
    opt=optim.Adam(model.parameters(), lr=args.lr)
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for ep in range(1,args.epochs+1):
        model.train(); sums={k:0.0 for k in ['loss','recon','codebook','commit']}
        for x,_ in tr:
            x=x.to(dev); xb=renorm(x,dev)
            xr,cb,cm=model(xb)
            L=model.loss_fn(xb,xr,cb,cm)
            opt.zero_grad(set_to_none=True); L['loss'].backward(); opt.step()
            for k in sums: sums[k]+=L[k].item()
        n=len(tr); print(f"[Epoch {ep}] loss={sums['loss']/n:.4f} recon={sums['recon']/n:.4f} codebook={sums['codebook']/n:.4f} commit={sums['commit']/n:.4f}")
        with torch.no_grad():
            x,_=next(iter(te)); x=x.to(dev); xb=renorm(x,dev); xr,_,_=model(xb)
            grid=torchvision.utils.make_grid(torch.cat([xb[:8],xr[:8]],0), nrow=8)
            torchvision.utils.save_image(grid, out/f"recon_epoch{ep:03d}.png")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='./data')
    ap.add_argument('--out', type=str, default='./runs/vq_vae')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--latent', type=int, default=64)
    ap.add_argument('--codes', type=int, default=512)
    ap.add_argument('--beta_c', type=float, default=0.25)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--cpu', action='store_true')
    args=ap.parse_args(); train(args)
