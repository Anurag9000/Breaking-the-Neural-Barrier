import argparse, os, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from _common_real_image import infer_num_classes, make_real_image_loaders
from model_pvt_v1 import PVTv1


def seed_all(s=42):
    random.seed(s); os.environ['PYTHONHASHSEED']=str(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def loaders(dataset, root, img, batch, workers, seed=42):
    return make_real_image_loaders(root, batch, num_workers=workers, image_size=img)


def train(m,dl,dev,crit,opt):
    m.train(); s=0.0
    for x,y in dl:
        x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True)
        o=m(x); L=crit(o,y); L.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); s+=L.item()*x.size(0)
    return s/len(dl.dataset)

def evaluate(m,dl,dev,crit):
    m.eval(); s=0.0; c=0
    with torch.no_grad():
        for x,y in dl:
            x,y=x.to(dev),y.to(dev); o=m(x); L=crit(o,y); s+=L.item()*x.size(0); c+=(o.argmax(1)==y).sum().item()
    return s/len(dl.dataset), c/len(dl.dataset)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='imagefolder',choices=['imagefolder'])
    p.add_argument('--data-root',default='./data')
    p.add_argument('--img-size',type=int,default=224)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--patience',type=int,default=25)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--wd',type=float,default=0.05)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save',default='PVTv1_best.pth')
    a=p.parse_args()

    seed_all(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tl,vl,te=loaders(a.dataset,a.data_root,a.img_size,a.batch_size,a.workers,a.seed)
    nc=infer_num_classes(tl)
    m=PVTv1(num_classes=nc).to(dev)
    crit=nn.CrossEntropyLoss(); opt=optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.wd)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)

    best=float('inf'); bp=None; bad=0
    for e in range(1,a.epochs+1):
        tr=train(m,tl,dev,crit,opt); vl,va=evaluate(m,vl,dev,crit); sch.step()
        print(f'Epoch {e:03d} | tr {tr:.4f} | val {vl:.4f} | acc {va*100:.2f}%')
        if vl<best-1e-4: best=vl; bp={k:v.cpu() for k,v in m.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=a.patience: print('Early stopping.'); break
    if bp: m.load_state_dict(bp)
    torch.save(m.state_dict(), a.save)
    tloss,tacc=evaluate(m,te,dev,crit); print(f'TEST | loss {tloss:.4f} | acc {tacc*100:.2f}%')

if __name__=='__main__':
    main()
