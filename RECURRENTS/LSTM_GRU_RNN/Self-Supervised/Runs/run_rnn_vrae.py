import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rnn_vrae import VRAEGRU
from _common_forda import make_forda_sequence_loaders

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience=patience; self.min_delta=min_delta; self.best=float('inf'); self.count=0
    def step(self,v):
        if v<self.best-self.min_delta: self.best=v; self.count=0; return True
        self.count+=1; return False
    def should_stop(self): return self.count>=self.patience


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--hidden',type=int,default=256)
    ap.add_argument('--latent',type=int,default=128)
    ap.add_argument('--layers',type=int,default=1)
    ap.add_argument('--T',type=int,default=64)
    ap.add_argument('--D',type=int,default=16)
    ap.add_argument('--n',type=int,default=40000)
    ap.add_argument('--val_split',type=float,default=0.1)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--beta',type=float,default=1.0)
    ap.add_argument('--save',type=str,default='vrae_gru_best.pt')
    args=ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    tr, va, _, _ = make_forda_sequence_loaders(batch_size=args.batch, seed=args.seed, return_index=False)

    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=VRAEGRU(1,args.hidden,args.latent,args.layers).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr)
    rec_loss=nn.MSELoss()
    es=EarlyStopper(args.patience,1e-4)

    best=None
    for ep in range(1,args.epochs+1):
        net.train(); trl=0.0
        for x in tr:
            x=x.transpose(1, 2).to(dev)
            dec_inp=torch.zeros_like(x); dec_inp[:,1:,:]=x[:,:-1,:]
            y,mu,lv=net(x,dec_inp)
            loss=rec_loss(y,x)+args.beta*VRAEGRU.kld(mu,lv)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            trl+=loss.item()*x.size(0)
        trl/=len(tr.dataset)

        net.eval(); val=0.0
        with torch.no_grad():
            for x in va:
                x=x.transpose(1, 2).to(dev)
                dec_inp=torch.zeros_like(x); dec_inp[:,1:,:]=x[:,:-1,:]
                y,mu,lv=net(x,dec_inp)
                loss=rec_loss(y,x)+args.beta*VRAEGRU.kld(mu,lv)
                val+=loss.item()*x.size(0)
        val/=len(va.dataset)

        if es.step(val): best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        print(f'Epoch {ep:03d} | train {trl:.6f} | val {val:.6f} | best {es.best:.6f}')
        if es.should_stop(): print('Early stopping.'); break

    if best is not None: net.load_state_dict(best)
    torch.save(net.state_dict(),args.save)
    print(f'Saved best model to {args.save}')

if __name__=='__main__':
    main()
