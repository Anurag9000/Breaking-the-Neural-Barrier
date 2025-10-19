import argparse, os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from vae_semi_m2 import VAE_M2, M2Config


def loaders(root, bs, workers):
    tf_tr = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    tf_ev = transforms.Compose([transforms.ToTensor()])
    full = datasets.CIFAR10(root=root, train=True, download=True, transform=tf_tr)
    n_tr = int(0.9*len(full)); n_val = len(full)-n_tr
    tr_set, _ = random_split(full, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    full_ev = datasets.CIFAR10(root=root, train=True, download=False, transform=tf_ev)
    _, val_set = random_split(full_ev, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    return (
        DataLoader(tr_set, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True),
    )

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); tot=recon_t=kl_t=ce_t=0.0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        xh, mu, lv, cls = model(x, y)
        loss, r, kl, ce = model.loss(x, xh, mu, lv, cls, y)
        bs=x.size(0); tot+=loss.item()*bs; recon_t+=r.item()*bs; kl_t+=kl.item()*bs; ce_t+=ce.item()*bs
    n=len(loader.dataset); return tot/n, recon_t/n, kl_t/n, ce_t/n


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    tr, va = loaders(args.data, args.batch_size, args.workers)
    cfg = M2Config(width=args.width, latent_dim=args.latent, tau=args.tau)
    model = VAE_M2(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best=float('inf'); best_state=None; bad=0
    for ep in range(1, args.epochs+1):
        model.train(); run=0.0
        for x,y in tr:
            x,y=x.to(device),y.to(device)
            xh, mu, lv, cls = model(x, y)
            loss, *_ = model.loss(x, xh, mu, lv, cls, y)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            run+=loss.item()*x.size(0)
        run/=len(tr.dataset)
        vloss, vrecon, vkl, vce = evaluate(model, va, device)
        if vloss+1e-12<best:
            best=vloss; best_state={'model':model.state_dict(),'cfg':cfg.__dict__,'epoch':ep}; bad=0
        else: bad+=1
        if args.verbose:
            print(f"ep {ep:03d} train {run:.4f} | val {vloss:.4f} (recon {vrecon:.4f} kl {vkl:.4f} ce {vce:.4f})")
        if bad>=args.patience:
            if args.verbose: print('early stop'); break

    if best_state:
        os.makedirs(args.out, exist_ok=True)
        torch.save(best_state, os.path.join(args.out,'vae_semi_m2_cifar10.pth'))
        with open(os.path.join(args.out,'vae_semi_m2_metrics.json'),'w') as f:
            json.dump({'best_val':best},f,indent=2)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=str,default='./data')
    ap.add_argument('--out',type=str,default='./artifacts')
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--width',type=int,default=128)
    ap.add_argument('--latent',type=int,default=32)
    ap.add_argument('--tau',type=float,default=0.67)
    ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--wd',type=float,default=1e-4)
    ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--verbose',action='store_true')
    args=ap.parse_args(); main(args)
