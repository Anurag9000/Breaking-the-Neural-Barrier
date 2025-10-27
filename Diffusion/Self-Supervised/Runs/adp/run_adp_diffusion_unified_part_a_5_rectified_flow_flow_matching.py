"""
Runner for ADP Diffusion – Part A5 (Rectified Flow / Flow Matching)
Flags:
  --family rfm
  --adp {w2d,d2w,alt_w,alt_d,depth_only,width_only}
  --bridge {linear,ve}  --sigma <float> (for ve)

Example (smoke):
python run_adp_diffusion_rfm.py --family rfm --adp w2d --bridge linear --smoke
"""
import argparse
from pathlib import Path
import torch
import torchvision as tv
import torchvision.transforms as T

from adp_diffusion_unified_rfm_model import (
    FlowCfg, TrainCfg, SearchCfg, build_model, ADP_REGISTRY
)


def build_loaders(data_root: str, batch_size: int, val_split=0.1, num_workers=2, image_size=32):
    tf_train = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(image_size, padding=4), T.ToTensor()])
    tf_eval  = T.Compose([T.ToTensor()])
    train_full = tv.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    eval_full  = tv.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_eval)
    n = len(train_full); nval = int(n*val_split)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(1234))
    val_idx, tr_idx = idx[:nval], idx[nval:]
    train = torch.utils.data.Subset(train_full, tr_idx)
    val   = torch.utils.data.Subset(eval_full,  val_idx)
    tr_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    va_loader = torch.utils.data.DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tr_loader, va_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', type=str, default='rfm', choices=['rfm'])
    p.add_argument('--adp', type=str, required=True, choices=list(ADP_REGISTRY.keys()))

    # flow cfg
    p.add_argument('--bridge', type=str, default='linear', choices=['linear','ve'])
    p.add_argument('--sigma', type=float, default=1.0)

    # model
    p.add_argument('--base', type=int, default=32)
    p.add_argument('--stages', type=int, default=3)
    p.add_argument('--blocks', type=int, default=1)

    # train
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--data-root', type=str, default='./data')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--wd', type=float, default=1e-4)

    # search
    p.add_argument('--delta', type=float, default=1e-4)
    p.add_argument('--trials-depth', type=int, default=2)
    p.add_argument('--trials-width', type=int, default=2)
    p.add_argument('--max-depth', type=int, default=8)
    p.add_argument('--max-base', type=int, default=192)

    args = p.parse_args()

    tr_loader, va_loader = build_loaders(args.data_root, args.batch_size, num_workers=args.num_workers)
    model = build_model(in_channels=3, base=args.base, stages=args.stages, blocks=args.blocks, out_channels=3)

    max_epochs = 2 if args.smoke else args.max_epochs
    patience = 1 if args.smoke else args.patience
    tcfg = TrainCfg(max_epochs=max_epochs, patience=patience, lr=args.lr, weight_decay=args.wd)
    fcfg = FlowCfg(bridge=args.bridge, sigma=args.sigma)
    scfg = SearchCfg(max_depth=args.max_depth, max_base=args.max_base, trials_width=args.trials_width, trials_depth=args.trials_depth, delta=args.delta)

    adp = ADP_REGISTRY[args.adp]
    model, best_val = adp(model, tr_loader, va_loader, tcfg, fcfg, scfg)

    out = Path('results_adp_diffusion'); out.mkdir(parents=True, exist_ok=True)
    tag = f"rfm_{args.bridge}_sig{args.sigma}_{args.adp}_b{args.base}_k{args.blocks}"
    torch.save({'model': model.state_dict(), 'best_val': best_val, 'args': vars(args)}, out / f'{tag}.pth')
    print({'best_val': best_val, 'params': sum(p.numel() for p in model.parameters())})

if __name__ == '__main__':
    main()
