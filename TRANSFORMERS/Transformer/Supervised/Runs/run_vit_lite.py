import argparse, os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from _common_real_image import infer_num_classes, make_real_image_loaders
from model_vit_lite import ViT_Lite


def set_seed(seed=42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_loaders(dataset, data_root, img_size, batch, workers, seed=42):
    return make_real_image_loaders(data_root, batch_size=batch, num_workers=workers, image_size=img_size)


def train_ep(m, dl, dev, crit, opt):
    m.train(); s=0.0
    for x,y in dl:
        x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True)
        o=m(x); L=crit(o,y); L.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        s+=L.item()*x.size(0)
    return s/len(dl.dataset)


def eval_ep(m, dl, dev, crit):
    m.eval(); s=0.0; c=0
    with torch.no_grad():
        for x,y in dl:
            x,y=x.to(dev),y.to(dev); o=m(x); L=crit(o,y); s+=L.item()*x.size(0); c+=(o.argmax(1)==y).sum().item()
    return s/len(dl.dataset), c/len(dl.dataset)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='imagefolder',choices=['imagefolder'])
    p.add_argument('--data-root',default='./data')
    p.add_argument('--img-size',type=int,default=160)
    p.add_argument('--patch',type=int,default=8)
    p.add_argument('--embed',type=int,default=192)
    p.add_argument('--depth',type=int,default=12)
    p.add_argument('--heads',type=int,default=3)
    p.add_argument('--mlp-ratio',type=float,default=3.0)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--wd',type=float,default=0.05)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save',default='ViT_Lite_best.pth')
    a=p.parse_args()

    set_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tl,vl,te=get_loaders(a.dataset,a.data_root,a.img_size,a.batch_size,a.workers,a.seed)
    nc=infer_num_classes(tl)
    m=ViT_Lite(img_size=a.img_size,patch_size=a.patch,num_classes=nc,embed_dim=a.embed,depth=a.depth,num_heads=a.heads,mlp_ratio=a.mlp_ratio).to(dev)
    crit=nn.CrossEntropyLoss(); opt=optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.wd)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)

    best=float('inf'); best_state=None; bad=0
    for e in range(1,a.epochs+1):
        tr=train_ep(m,tl,dev,crit,opt); vloss,vacc=eval_ep(m,vl,dev,crit); sch.step()
        print(f'Epoch {e:03d} | tr {tr:.4f} | val {vloss:.4f} | acc {vacc*100:.2f}%')
        if vloss<best-1e-4:
            best=vloss; best_state={k:v.cpu() for k,v in m.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=a.patience: print('Early stopping.'); break
    if best_state: m.load_state_dict(best_state)
    torch.save(m.state_dict(),a.save)
    tloss,tacc=eval_ep(m,te,dev,crit); print(f'TEST | loss {tloss:.4f} | acc {tacc*100:.2f}%')

if __name__=='__main__':
    main()
