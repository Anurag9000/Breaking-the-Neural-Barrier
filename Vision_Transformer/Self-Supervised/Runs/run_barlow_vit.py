"""Runner: Barlow Twins ViT"""
import argparse, torch, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from barlow_vit import BarlowTwinsViT, BarlowConfig

class TwoCrops:
    def __init__(self, base): self.base=base
    def __call__(self, x): return self.base(x), self.base(x)

def loaders(dataset, data_dir, batch, workers=2):
    norm = transforms.Normalize((0.4914,0.4822,0.4465) if dataset=='cifar10' else (0.5071,0.4867,0.4408),
                                (0.2470,0.2435,0.2616) if dataset=='cifar10' else (0.2675,0.2565,0.2761))
    base = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(), norm])
    aug=TwoCrops(base)
    plain = transforms.Compose([transforms.ToTensor(), norm])
    if dataset=='cifar10':
        full=datasets.CIFAR10(data_dir, train=True, transform=aug, download=True)
        test=datasets.CIFAR10(data_dir, train=False, transform=plain, download=True)
    else:
        full=datasets.CIFAR100(data_dir, train=True, transform=aug, download=True)
        test=datasets.CIFAR100(data_dir, train=False, transform=plain, download=True)
    val_size=5000; tr_size=len(full)-val_size
    tr,va=random_split(full,[tr_size,val_size],generator=torch.Generator().manual_seed(42))
    return DataLoader(tr,batch,True,num_workers=workers,pin_memory=True), \
           DataLoader(va,batch,False,num_workers=workers,pin_memory=True), \
           DataLoader(test,batch,False,num_workers=workers,pin_memory=True)


def step(model, batch, device, opt=None):
    (x1,x2), _ = batch
    x1=x1.to(device); x2=x2.to(device)
    loss, logs = model(x1,x2)
    if opt is not None:
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return loss.item(), logs

@torch.no_grad()
def run_epoch(model, loader, device, train=False, opt=None):
    model.train(train); total=0.0
    for b in loader:
        l, _ = step(model,b,device,opt if train else None)
        total += l * (b[0][0].size(0))
    return total/len(loader.dataset)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10', choices=['cifar10','cifar100'])
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-6)
    ap.add_argument('--img_size', type=int, default=32)
    ap.add_argument('--patch_size', type=int, default=4)
    ap.add_argument('--dim', type=int, default=384)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--mlp_ratio', type=float, default=4.0)
    ap.add_argument('--proj_dim', type=int, default=8192)
    ap.add_argument('--proj_hidden', type=int, default=4096)
    ap.add_argument('--lambd', type=float, default=0.0051)
    ap.add_argument('--save', default='barlow_vit_best.pt')
    args=ap.parse_args()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr,va,te=loaders(args.dataset,args.data_dir,args.batch_size)

    cfg=BarlowConfig(args.img_size,args.patch_size,args.dim,args.depth,args.heads,args.mlp_ratio,
                     args.proj_dim,args.proj_hidden,args.lambd)
    model=BarlowTwinsViT(cfg).to(device)
    opt=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best=float('inf'); best_state=None; bad=0
    for ep in range(1,args.epochs+1):
        trl=run_epoch(model,tr,device,True,opt)
        val=run_epoch(model,va,device,False)
        print(f"epoch {ep:03d} | train {trl:.4f} | val {val:.4f}")
        if val+1e-6<best: best=val; best_state={k:v.cpu() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=args.patience: print('Early stopping.'); break
    if best_state:
        model.load_state_dict(best_state); torch.save(best_state,args.save)
        print(f"Saved {args.save} (val {best:.4f})")
    test=run_epoch(model,te,device,False)
    print(f"Test objective: {test:.4f}")

if __name__=='__main__':
    main()
