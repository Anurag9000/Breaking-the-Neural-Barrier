import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as T

from adp_vae_vmf import VMFVAEConfig, VMFVAE

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
    cfg=VMFVAEConfig(latent_dim=args.latent, kappa_min=args.kappa_min, width=args.width, depth=args.depth)
    model=VMFVAE(cfg).to(dev)
    opt=optim.Adam(model.parameters(), lr=args.lr)
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for ep in range(1,args.epochs+1):
        model.train(); sums={k:0.0 for k in ['loss','recon','kl']}
        for x,_ in tr:
            x=x.to(dev); xb=renorm(x,dev)
            xr,mu,kappa,z=model(xb)
            L=model.loss_fn(xb,xr,mu,kappa,z)
            opt.zero_grad(set_to_none=True); L['loss'].backward(); opt.step()
            for k in sums: sums[k]+=L[k].item()
        n=len(tr); print(f"[Epoch {ep}] loss={sums['loss']/n:.4f} recon={sums['recon']/n:.4f} kl={sums['kl']/n:.4f}")
        with torch.no_grad():
            x,_=next(iter(te)); x=x.to(dev); xb=renorm(x,dev); xr,_,_,_=model(xb)
            grid=torchvision.utils.make_grid(torch.cat([xb[:8],xr[:8]],0), nrow=8)
            torchvision.utils.save_image(grid, out/f"recon_epoch{ep:03d}.png")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='./data')
    ap.add_argument('--out', type=str, default='./runs/vae_vmf')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--latent', type=int, default=16)
    ap.add_argument('--kappa_min', type=float, default=1.0)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--cpu', action='store_true')
    args=ap.parse_args(); train(args)
