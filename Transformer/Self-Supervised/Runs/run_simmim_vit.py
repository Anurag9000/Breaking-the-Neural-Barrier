import os, json, argparse, random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from dataclasses import dataclass
from model_simmim_vit import SimMIMViT

@dataclass
class Config:
    data_root: str = './data'
    dataset: str = 'CIFAR10'
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.6
    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.05
    patience: int = 20
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir: str = './outs_simmim_vit'
    seed: int = 42


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(cfg: Config):
    size = cfg.img_size
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
    ])
    if cfg.dataset == 'CIFAR10':
        full = datasets.CIFAR10(cfg.data_root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(cfg.data_root, train=False, download=True, transform=eval_tf)
    else:
        full = datasets.CIFAR100(cfg.data_root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR100(cfg.data_root, train=False, download=True, transform=eval_tf)
    n_val = int(0.1 * len(full)); n_train = len(full) - n_val
    g = torch.Generator().manual_seed(cfg.seed)
    train, val = random_split(full, [n_train, n_val], generator=g)
    val.dataset.transform = eval_tf
    return (DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True),
            DataLoader(test, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True))


def save_plot(curve, title, path, semilogy=False):
    import matplotlib.pyplot as plt
    plt.figure(); plt.semilogy(curve) if semilogy else plt.plot(curve)
    plt.title(title); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight'); plt.close()


def train(cfg: Config):
    set_seed(cfg.seed)
    train_loader, val_loader, test_loader = make_loaders(cfg)
    model = SimMIMViT(cfg.img_size, cfg.patch_size, 3, cfg.embed_dim, cfg.depth, cfg.heads, cfg.mlp_ratio, cfg.mask_ratio).to(cfg.device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val=float('inf'); best_state=None; patience=cfg.patience
    tr_curve=[]; va_curve=[]

    for ep in range(cfg.epochs):
        model.train(); tr=0.0
        for imgs,_ in train_loader:
            imgs=imgs.to(cfg.device)
            pred, mask = model(imgs)
            loss = model.loss((pred, mask), imgs)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tr += loss.item()*imgs.size(0)
        tr/=len(train_loader.dataset); tr_curve.append(tr)

        model.eval(); va=0.0
        with torch.no_grad():
            for imgs,_ in val_loader:
                imgs=imgs.to(cfg.device)
                pred, mask = model(imgs)
                loss = model.loss((pred, mask), imgs)
                va += loss.item()*imgs.size(0)
        va/=len(val_loader.dataset); va_curve.append(va)
        print(f'Epoch {ep+1}/{cfg.epochs} | train {tr:.4f} | val {va:.4f}')
        if va < best_val - 1e-4:
            best_val=va; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; patience=cfg.patience
        else:
            patience-=1
            if patience==0:
                print('Early stopping.'); break

    os.makedirs(cfg.out_dir, exist_ok=True)
    save_plot(tr_curve, 'SimMIM Train Loss', os.path.join(cfg.out_dir,'train_loss.png'), semilogy=True)
    save_plot(va_curve, 'SimMIM Val Loss', os.path.join(cfg.out_dir,'val_loss.png'), semilogy=True)
    if best_state is not None:
        torch.save(best_state, os.path.join(cfg.out_dir,'best_state.pt'))
        with open(os.path.join(cfg.out_dir,'summary.json'),'w') as f:
            json.dump({'best_val': float(best_val)}, f, indent=2)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--dataset', type=str, default='CIFAR10')
    a.add_argument('--img_size', type=int, default=224)
    a.add_argument('--patch_size', type=int, default=16)
    a.add_argument('--embed_dim', type=int, default=384)
    a.add_argument('--depth', type=int, default=6)
    a.add_argument('--heads', type=int, default=6)
    a.add_argument('--mlp_ratio', type=float, default=4.0)
    a.add_argument('--mask_ratio', type=float, default=0.6)
    a.add_argument('--batch_size', type=int, default=128)
    a.add_argument('--epochs', type=int, default=200)
    a.add_argument('--lr', type=float, default=1e-3)
    a.add_argument('--weight_decay', type=float, default=0.05)
    a.add_argument('--patience', type=int, default=20)
    a.add_argument('--out_dir', type=str, default='./outs_simmim_vit')
    a.add_argument('--seed', type=int, default=42)
    cfg = Config(**vars(a.parse_args()))
    train(cfg)
