import argparse, os, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model_swin import Swin


def seed_all(s=42):
    random.seed(s); os.environ['PYTHONHASHSEED']=str(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_loaders(dataset, root, img, batch, workers, seed=42):
    norm=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    tr=transforms.Compose([transforms.Resize((img,img)), transforms.RandomResizedCrop(img,scale=(0.8,1.0)),
                           transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    ev=transforms.Compose([transforms.Resize((img,img)), transforms.ToTensor(), norm])
    DS=datasets.CIFAR10 if dataset=='cifar10' else datasets.CIFAR100
    A=DS(root=root,train=True,download=True,transform=tr)
    B=DS(root=root,train=True,download=True,transform=ev)
    T=DS(root=root,train=False,download=True,transform=ev)
    n=int(0.9*len(A)); g=torch.Generator().manual_seed(seed)
    trS,_=random_split(A,[n,len(A)-n],generator=g); _,vaS=random_split(B,[n,len(B)-n],generator=g)
    return DataLoader(trS,batch,True,num_workers=workers,pin_memory=True), DataLoader(vaS,batch,False,num_workers=workers,pin_memory=True), DataLoader(T,batch,False,num_workers=workers,pin_memory=True)


def train_ep(m,dl,dev,crit,opt):
    m.train(); s=0.0
    for x,y in dl:
        x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True)
        o=m(x); L=crit(o,y); L.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); s+=L.item()*x.size(0)
    return s/len(dl.dataset)

def eval_ep(m,dl,dev,crit):
    m.eval(); s=0.0; c=0
    with torch.no_grad():
        for x,y in dl:
            x,y=x.to(dev),y.to(dev); o=m(x); L=crit(o,y); s+=L.item()*x.size(0); c+=(o.argmax(1)==y).sum().item()
    return s/len(dl.dataset), c/len(dl.dataset)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100'])
    p.add_argument('--data-root',default='./data')
    p.add_argument('--img-size',type=int,default=224)
    p.add_argument('--embed',type=int,default=96)
    p.add_argument('--depths',type=int,nargs='+',default=[2,2,6,2])
    p.add_argument('--heads',type=int,nargs='+',default=[3,6,12,24])
    p.add_argument('--win',type=int,default=7)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--patience',type=int,default=25)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--wd',type=float,default=0.05)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save',default='Swin_best.pth')
    a=p.parse_args()

    seed_all(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tl,vl,te=get_loaders(a.dataset,a.data_root,a.img_size,a.batch_size,a.workers,a.seed)
    nc=10 if a.dataset=='cifar10' else 100
    m=Swin(img_size=a.img_size,num_classes=nc,embed_dim=a.embed,depths=tuple(a.depths),heads=tuple(a.heads),win=a.win).to(dev)

    crit=nn.CrossEntropyLoss(); opt=optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.wd)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)

    best=float('inf'); bp=None; bad=0
    for e in range(1,a.epochs+1):
        tr=train_ep(m,tl,dev,crit,opt); vl,va=eval_ep(m,vl,dev,crit); sch.step()
        print(f'Epoch {e:03d} | tr {tr:.4f} | val {vl:.4f} | acc {va*100:.2f}%')
        if vl<best-1e-4: best=vl; bp={k:v.cpu() for k,v in m.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=a.patience: print('Early stopping.'); break
    if bp: m.load_state_dict(bp)
    torch.save(m.state_dict(), a.save)
    tloss,tacc=eval_ep(m,te,dev,crit); print(f'TEST | loss {tloss:.4f} | acc {tacc*100:.2f}%')

if __name__=='__main__':
    main()
