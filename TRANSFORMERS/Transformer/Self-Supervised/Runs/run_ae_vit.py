"""Runner: Plain ViT Autoencoder"""
import argparse, torch, torch.optim as optim
from ae_vit import ViTAE, AEConfig
from _common_real_image import make_real_image_loaders


def loaders(dataset, data_dir, batch, workers=0):
    return make_real_image_loaders(data_dir, batch, image_size=32, num_workers=workers)


def train_epoch(model, loader, device, opt):
    model.train(); total=0.0
    for x,_ in loader:
        x=x.to(device)
        loss,_=model(x)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        total+=loss.item()*x.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval(); total=0.0
    for x,_ in loader:
        x=x.to(device)
        loss,_=model(x); total+=loss.item()*x.size(0)
    return total/len(loader.dataset)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', default='imagefolder')
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--img', type=int, default=32)
    ap.add_argument('--patch', type=int, default=4)
    ap.add_argument('--dim', type=int, default=384)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--ratio', type=float, default=4.0)
    ap.add_argument('--save', default='ae_vit_best.pt')
    args=ap.parse_args()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr,va,te=loaders(args.dataset,args.data_dir,args.batch_size)

    cfg=AEConfig(args.img,args.patch,args.dim,args.depth,args.heads,args.ratio)
    model=ViTAE(cfg).to(device)
    opt=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best=float('inf'); best_state=None; bad=0
    for ep in range(1,args.epochs+1):
        trl=train_epoch(model,tr,device,opt)
        val=eval_epoch(model,va,device)
        print(f"epoch {ep:03d} | train {trl:.4f} | val {val:.4f}")
        if val+1e-6<best: best=val; best_state={k:v.cpu() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=args.patience: print('Early stopping.'); break
    if best_state:
        model.load_state_dict(best_state); torch.save(best_state,args.save)
        print(f"Saved {args.save} (val {best:.4f})")
    test=eval_epoch(model,te,device)
    print(f"Test reconstruction loss: {test:.4f}")

if __name__=='__main__':
    main()
