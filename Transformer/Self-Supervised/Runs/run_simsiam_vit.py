import os, json, argparse, random
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
sys.path.append(str(Path(__file__).resolve().parent))
from utils.adp_logging import ContinuousLogger
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
from model_simsiam_vit import SimSiamViT
from _common_real_image import make_two_crops_loaders

@dataclass
class Config:
    data_root: str = './data'
    dataset: str = 'imagefolder'
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    proj_hidden: int = 2048
    proj_out: int = 2048
    pred_hidden: int = 512
    batch_size: int = 256
    epochs: int = 400
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 30
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir: str = './outs_simsiam_vit'
    seed: int = 42


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(cfg: Config):
    return make_two_crops_loaders(cfg.data_root, cfg.batch_size, image_size=cfg.img_size, num_workers=4)


def save_plot(curve, title, path, semilogy=False):
    plt.figure(); plt.semilogy(curve) if semilogy else plt.plot(curve)
    plt.title(title); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight'); plt.close()


def train(cfg: Config):
    set_seed(cfg.seed)
    train_loader, val_loader, _ = make_loaders(cfg)
    model = SimSiamViT(cfg.img_size, cfg.patch_size, cfg.embed_dim, cfg.depth, cfg.heads, cfg.mlp_ratio,
                       cfg.proj_hidden, cfg.proj_out, cfg.pred_hidden).to(cfg.device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float('inf'); best_state=None; patience = cfg.patience
    tr_curve=[]; va_curve=[]


    # Init Logger


    logger = ContinuousLogger(Path('results_run_simsiam_vit'), 'run_simsiam_vit', 'train')


    for epoch in range(cfg.epochs):
        model.train(); tr_loss=0
        for imgs, _ in train_loader:
            x1 = imgs.to(cfg.device)
            x2 = torch.stack([train_loader.dataset.dataset.transform(img) for img in imgs])
            x2 = x2.to(cfg.device)
            loss = model(x1, x2)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_loss += loss.item() * imgs.size(0)
        tr_loss /= len(train_loader.dataset); tr_curve.append(tr_loss)

        model.eval(); va_loss=0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(cfg.device)
                loss = model(imgs, imgs)
                va_loss += loss.item() * imgs.size(0)
        va_loss /= len(val_loader.dataset); va_curve.append(va_loss)

        # Log


        msg = f"Epoch {epoch+1}/{cfg.epochs} | train {tr_loss:.4f} | val {va_loss:.4f}"


        logger.log_console(msg)


        logger.log_epoch_stats({


            "epoch": epoch,


            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),


            "train_loss": loss.item() if 'loss' in locals() else 0


        })
        if va_loss < best_val - 1e-4:
            best_val = va_loss; best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}; patience = cfg.patience
        else:
            patience -= 1
            if patience==0:
                print('Early stopping.'); break

    os.makedirs(cfg.out_dir, exist_ok=True)
    save_plot(tr_curve, 'SimSiam Train Loss', os.path.join(cfg.out_dir,'train_loss.png'), semilogy=True)
    save_plot(va_curve, 'SimSiam Val Loss', os.path.join(cfg.out_dir,'val_loss.png'), semilogy=True)

    if best_state is not None:
        torch.save(best_state, os.path.join(cfg.out_dir, 'best_state.pt'))
        with open(os.path.join(cfg.out_dir,'summary.json'),'w') as f:
            json.dump({'best_val': float(best_val)}, f, indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='imagefolder')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--embed_dim', type=int, default=384)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--heads', type=int, default=6)
    p.add_argument('--mlp_ratio', type=float, default=4.0)
    p.add_argument('--proj_hidden', type=int, default=2048)
    p.add_argument('--proj_out', type=int, default=2048)
    p.add_argument('--pred_hidden', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--out_dir', type=str, default='./outs_simsiam_vit')
    p.add_argument('--seed', type=int, default=42)
    cfg = Config(**vars(p.parse_args()))
    train(cfg)
