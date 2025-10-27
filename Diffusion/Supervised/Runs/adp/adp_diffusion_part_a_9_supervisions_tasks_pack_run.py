# ============================================================
# File: run_adp_diff_tasks.py  (RUN)
# Runner for multi-task DDPM (ε-pred) with 6 ADP policies
# Tasks: --task {inpaint, sr, control, translate, segcond, regress}
# ============================================================

import argparse
import random

import torch
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from adp_diff_tasks import TaskDDPMSingleModel, TrainCfg, SearchCfg, POLICIES

# -----------------------------
# Utility: random box masks for inpainting
# -----------------------------

def random_box_mask(B, H, W, min_frac=0.25, max_frac=0.6, device='cpu'):
    m = torch.zeros(B, 1, H, W, device=device)
    for b in range(B):
        fh = random.uniform(min_frac, max_frac)
        fw = random.uniform(min_frac, max_frac)
        h = max(1, int(H * fh)); w = max(1, int(W * fw))
        y0 = random.randint(0, max(0, H - h)); x0 = random.randint(0, max(0, W - w))
        m[b, :, y0:y0+h, x0:x0+w] = 1.0
    return m

# -----------------------------
# Conditioning builders per task
# -----------------------------

def build_cond(task, x, img_size, seg_classes=0, sr_scale=4, aux_reg_dim=0):
    if task == 'inpaint':
        B, C, H, W = x.shape
        mask = random_box_mask(B, H, W, device=x.device)
        hint = x * (1.0 - mask)
        cond = torch.cat([hint, mask], dim=1)  # C+1 channels
        return cond, None
    if task == 'sr':
        # create LR by downsampling & upsampling (bicubic)
        lr = torch.nn.functional.interpolate(x, scale_factor=1.0/sr_scale, mode='bicubic', align_corners=False, recompute_scale_factor=True)
        lr_up = torch.nn.functional.interpolate(lr, size=x.shape[-2:], mode='bicubic', align_corners=False)
        cond = lr_up
        return cond, None
    if task == 'control':
        # cheap edge map via Sobel-like kernels
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
        gray = x.mean(dim=1, keepdim=True)
        gx = torch.conv2d(gray, kx, padding=1)
        gy = torch.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy)
        cond = mag  # 1ch edge hint
        return cond, None
    if task == 'translate':
        # treat x as target; create a fake "source" by heavy blur + color jitter (proxy domain)
        blur = T.GaussianBlur(11, sigma=3.0)
        jitter = T.ColorJitter(0.5,0.5,0.5,0.2)
        # apply per-sample
        x_list = [x[i] for i in range(x.size(0))]
        src = torch.stack([jitter(blur(xi.cpu())).to(x.device) for xi in x_list], dim=0)
        cond = src
        return cond, None
    if task == 'segcond':
        # synth segmentation condition: k-means-ish colors to indices (toy). Here we just quantize to seg_classes bins per channel avg.
        if seg_classes <= 0: seg_classes = 8
        gray = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        bins = torch.clamp((gray + 1.0) * 0.5 * (seg_classes-1) + 0.5, 0, seg_classes-1).long()
        onehot = torch.zeros(x.size(0), seg_classes, x.size(2), x.size(3), device=x.device)
        onehot.scatter_(1, bins, 1.0)
        cond = onehot
        return cond, None
    if task == 'regress':
        # auxiliary regression target: predict per-image mean color (3-dim)
        aux = x.mean(dim=(2,3))  # (B,3)
        cond = torch.zeros_like(x)  # no external condition
        return cond, aux
    raise ValueError('Unknown task')


# -----------------------------
# Datasets
# -----------------------------

class CIFARCond(Dataset):
    def __init__(self, root, train, img_size, task, seg_classes, sr_scale, aux_reg_dim):
        tfm = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(img_size, padding=4) if train else T.Resize(img_size),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        self.base = tv.datasets.CIFAR10(root, train=train, download=True, transform=tfm)
        self.task = task; self.img_size = img_size
        self.seg_classes = seg_classes; self.sr_scale = sr_scale; self.aux_reg_dim = aux_reg_dim
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, _ = self.base[idx]
        cond, aux = build_cond(self.task, x.unsqueeze(0), self.img_size, self.seg_classes, self.sr_scale, self.aux_reg_dim)
        x = x; cond = cond[0]
        if aux is None:
            return x, cond
        else:
            return x, cond, aux[0]


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser(description='Multi-task DDPM (ε-pred) with ADP')

    p.add_argument('--task', type=str, default='inpaint', choices=['inpaint','sr','control','translate','segcond','regress'])
    p.add_argument('--adp', type=str, default='depth2width', choices=['depth2width','width2depth','alt_depth','alt_width','depth_only','width_only'])

    # Data
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--img-size', type=int, default=32)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--val-split', type=float, default=0.2)

    # Train cfg
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--max-epochs', type=int, default=30)
    p.add_argument('--es-patience', type=int, default=7)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Search cfg
    p.add_argument('--delta', type=float, default=0.0)
    p.add_argument('--trials-depth', type=int, default=20)
    p.add_argument('--trials-width', type=int, default=20)
    p.add_argument('--ex-k', type=int, default=8)
    p.add_argument('--max-neurons', type=int, default=0, help='0 disables capacity limit')

    # Model hyperparams
    p.add_argument('--widths', type=int, nargs='+', default=[32,64,96])
    p.add_argument('--T', type=int, default=1000)
    p.add_argument('--seg-classes', type=int, default=8)
    p.add_argument('--sr-scale', type=int, default=4)
    p.add_argument('--aux-reg-dim', type=int, default=0)
    p.add_argument('--lambda-aux', type=float, default=0.1)

    args = p.parse_args()

    # Determine cond channels per task
    if args.task == 'inpaint':
        cond_ch = 3 + 1
    elif args.task == 'sr':
        cond_ch = 3
    elif args.task == 'control':
        cond_ch = 1
    elif args.task == 'translate':
        cond_ch = 3
    elif args.task == 'segcond':
        cond_ch = args.seg_classes
    elif args.task == 'regress':
        cond_ch = 3  # dummy zero map
    else:
        raise ValueError('Unknown task')

    # Data loaders (CIFAR-10 based)
    full = CIFARCond(args.data, train=True, img_size=args.img_size, task=args.task, seg_classes=args.seg_classes, sr_scale=args.sr_scale, aux_reg_dim=args.aux_reg_dim)
    n = len(full); n_val = int(n * args.val_split)
    idx = torch.randperm(n); val_idx = idx[:n_val]; train_idx = idx[n_val:]
    train = torch.utils.data.Subset(full, train_idx.tolist())
    val   = torch.utils.data.Subset(full, val_idx.tolist())
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = TaskDDPMSingleModel(task=args.task, img_ch=3, cond_ch=cond_ch, widths=args.widths, T=args.T,
                                aux_reg_dim=args.aux_reg_dim, lambda_aux=args.lambda_aux)

    train_cfg = TrainCfg(lr=args.lr, max_epochs=args.max_epochs, es_patience=args.es_patience,
                         grad_clip=args.grad_clip, device=args.device)
    maxN = None if args.max_neurons == 0 else args.max_neurons
    search_cfg = SearchCfg(delta=args.delta, trials_width=args.trials_width, trials_depth=args.trials_depth,
                           ex_k=args.ex_k, max_neurons=maxN)

    best = POLICIES[args.adp](model, train_loader, val_loader, train_cfg, search_cfg)
    print(f"[TASK:{args.task}] Best val loss = {best:.4f}. Final neurons = {model.neurons()}.")


if __name__ == '__main__':
    main()

# ============================================================
# End of run_adp_diff_tasks.py
# ============================================================
