"""Runner: Jigsaw-ViT (3x3 tiles, K fixed permutations)"""
import argparse, torch, torch.optim as optim, random
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from jigsaw_vit import JigsawViT, JigsawConfig

# build a fixed set of K permutations of 9 tiles
def fixed_permutations(K, seed=123):
    random.seed(seed)
    perms=set()
    base=list(range(9))
    perms.add(tuple(base))
    while len(perms)<K:
        p=base[:]
        random.shuffle(p)
        perms.add(tuple(p))
    return [list(p) for p in list(perms)[:K]]

class JigsawWrapper:
    def __init__(self, base_tf, perms): self.base=base_tf; self.perms=perms
    def __call__(self, img):
        x=self.base(img)  # tensor CxHxW
        C,H,W=x.shape; gh=gw=3
        th, tw = H//gh, W//gw
        tiles=[x[:, i*th:(i+1)*th, j*tw:(j+1)*tw] for i in range(gh) for j in range(gw)]
        k=random.randint(0,len(self.perms)-1); p=self.perms[k]
        reordered=[tiles[idx] for idx in p]
        rows=[torch.cat(reordered[i*gw:(i+1)*gw], dim=2) for i in range(gh)]
        x_perm=torch.cat(rows, dim=1)
        return x_perm, k

class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, root, train, dataset, tf):
        if dataset=='cifar10': self.base=datasets.CIFAR10(root, train=train, download=True)
        else: self.base=datasets.CIFAR100(root, train=train, download=True)
        self.tf=tf
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img,_=self.base[idx]; return self.tf(img)


def loaders(dataset, data_dir, batch, K, workers=2):
    mean = (0.4914,0.4822,0.4465) if dataset=='cifar10' else (0.5071,0.4867,0.4408)
    std  = (0.2470,0.2435,0.2616) if dataset=='cifar10' else (0.2675,0.2565,0.2761)
    normalize = transforms.Normalize(mean, std)
    base = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize])
    perms=fixed_permutations(K)
    jig_tf = JigsawWrapper(base, perms)

    full=JigsawDataset(data_dir, True, dataset, jig_tf)
    val_size=5000; tr_size=len(full)-val_size
    tr,va=random_split(full,[tr_size,val_size],generator=torch.Generator().manual_seed(42))

    test_base = datasets.CIFAR10 if dataset=='cifar10' else datasets.CIFAR100
    test_raw = test_base(data_dir, train=False, download=True)
    test=[jig_tf(img) for img,_ in test_raw]

    def collate(batch):
        xs=[]; ys=[]
        for (x,k) in batch: xs.append(x); ys.append(torch.tensor(k))
        return torch.stack(xs,0), torch.stack(ys,0)

    return DataLoader(tr,batch,True,num_workers=workers,pin_memory=True,collate_fn=collate), \
           DataLoader(va,batch,False,num_workers=workers,pin_memory=True,collate_fn=collate), \
           DataLoader(test,batch,False,num_workers=workers,pin_memory=True,collate_fn=collate)


def train_epoch(model, loader, device, opt):
    model.train(); total=0.0
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        loss,_=model(x,y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        total+=loss.item()*x.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval(); total=0.0; correct=0
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        logits,_=model(x)
        loss=torch.nn.functional.cross_entropy(logits,y)
        total+=loss.item()*x.size(0)
        pred=logits.argmax(1); correct+=(pred==y).sum().item()
    return total/len(loader.dataset), correct/len(loader.dataset)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10', choices=['cifar10','cifar100'])
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-6)
    ap.add_argument('--img', type=int, default=32)
    ap.add_argument('--patch', type=int, default=4)
    ap.add_argument('--dim', type=int, default=384)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--ratio', type=float, default=4.0)
    ap.add_argument('--num_perms', type=int, default=30)
    ap.add_argument('--save', default='jigsaw_vit_best.pt')
    args=ap.parse_args()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr,va,te=loaders(args.dataset,args.data_dir,args.batch_size,args.num_perms)

    cfg=JigsawConfig(args.img,args.patch,args.dim,args.depth,args.heads,args.ratio,args.num_perms)
    model=JigsawViT(cfg).to(device)
    opt=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best=float('inf'); best_state=None; bad=0
    for ep in range(1,args.epochs+1):
        trl=train_epoch(model,tr,device,opt)
        val,acc=eval_epoch(model,va,device)
        print(f"epoch {ep:03d} | train {trl:.4f} | val {val:.4f} | val_acc {acc:.3f}")
        if val+1e-6<best: best=val; best_state={k:v.cpu() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=args.patience: print('Early stopping.'); break
    if best_state:
        model.load_state_dict(best_state); torch.save(best_state,args.save)
        print(f"Saved {args.save} (val {best:.4f})")
    test,acc=eval_epoch(model,te,device)
    print(f"Test pretext loss: {test:.4f} | acc {acc:.3f}")

if __name__=='__main__':
    main()
