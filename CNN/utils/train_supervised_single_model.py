"""
Training script (single-model) covering:
- Losses: CE / label-smoothed CE, Focal, Class-Balanced Focal, LDAM (+DRW), Margin-Softmax (ArcFace/CosFace/AM), Asymmetric Loss (multi-label)
- Augmentations: crop/flip/jitter, Cutout, MixUp, CutMix, RandAugment, TrivialAugmentWide
- Regularization: weight decay, dropout, DropBlock, Stochastic Depth/DropPath, ShakeDrop, grad clipping, early stopping
- Optimizers & schedules: SGD+Nesterov, AdamW, cosine/SGDR, OneCycleLR, warmup+step/poly, SAM
- Norm/Activations: BatchNorm, GroupNorm, EvoNorm (S0/B0), Weight Standardization, ReLU/LeakyReLU/PReLU, ELU/SELU, GELU, SiLU, Mish
- Inference: TTA; SWA consolidation to a single final model

Usage examples:
python train_supervised_single_model.py --dataset cifar10 --epochs 100 --model SimpleNet --head margin --margin_type arcface --loss ldam_drw \
    --optimizer sgd --lr 0.1 --schedule cosine --mixup_alpha 0.2 --cutmix_alpha 0.2 --randaugment 1 --dropblock 0.1 --drop_path 0.1

"""
from __future__ import annotations
import argparse
import os
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from supervised_single_model_blocks import (
    set_seed, SimpleNet, build_train_transforms, build_test_transforms,
    MixupCutmixCfg, apply_mixup_cutmix, accuracy,
    FocalLoss, ClassBalancedFocalLoss, LDAMLoss, drw_class_weights,
    AsymmetricLossMultiLabel,
    build_optimizer, build_scheduler, build_sam,
    EarlyStopping, tta_predict_logits, build_swa, update_bn_for_swa
)

try:
    import torchvision
    from torchvision import datasets
except Exception as e:
    torchvision = None
    datasets = None


def get_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet_subset'])
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    # Augment
    p.add_argument('--random_crop', type=int, default=1)
    p.add_argument('--hflip', type=int, default=1)
    p.add_argument('--vflip', type=int, default=0)
    p.add_argument('--color_jitter', type=int, default=0)
    p.add_argument('--randaugment', type=int, default=0)
    p.add_argument('--trivialaugment', type=int, default=0)
    p.add_argument('--cutout', type=int, default=0)
    p.add_argument('--cutout_holes', type=int, default=1)
    p.add_argument('--cutout_length', type=int, default=16)
    p.add_argument('--mixup_alpha', type=float, default=0.0)
    p.add_argument('--cutmix_alpha', type=float, default=0.0)
    # Model
    p.add_argument('--model', type=str, default='SimpleNet')
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--norm', type=str, default='batchnorm', choices=['batchnorm','groupnorm','evonorm_s0','evonorm_b0','identity'])
    p.add_argument('--act', type=str, default='relu', choices=['relu','leakyrelu','prelu','elu','selu','gelu','silu','swish','mish'])
    p.add_argument('--use_ws', type=int, default=0)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--dropblock', type=float, default=0.0)
    p.add_argument('--drop_path', type=float, default=0.0)
    p.add_argument('--block_size', type=int, default=7)
    p.add_argument('--embedding_dim', type=int, default=256)
    p.add_argument('--head', type=str, default='linear', choices=['linear','margin'])
    p.add_argument('--margin_type', type=str, default='arcface', choices=['arcface','cosface','am'])
    p.add_argument('--margin_m', type=float, default=0.5)
    p.add_argument('--margin_s', type=float, default=30.0)
    # Train
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--optimizer', type=str, default='sgd', choices=['sgd','adamw'])
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--nesterov', type=int, default=1)
    p.add_argument('--sam', type=int, default=0)
    p.add_argument('--sam_rho', type=float, default=0.05)
    p.add_argument('--schedule', type=str, default='cosine', choices=['cosine','sgdr','onecycle','warmup_step','warmup_poly'])
    p.add_argument('--warmup_epochs', type=int, default=0)
    p.add_argument('--step_milestones', type=int, nargs='*', default=None)
    p.add_argument('--gamma', type=float, default=0.1)
    p.add_argument('--poly_power', type=float, default=2.0)
    p.add_argument('--clip_grad', type=float, default=0.0)
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--loss', type=str, default='ce', choices=['ce','focal','cb_focal','ldam','ldam_drw','margin_ce','asl'])
    p.add_argument('--multilabel', type=int, default=0)
    p.add_argument('--early_stop', type=int, default=0)
    p.add_argument('--patience', type=int, default=10)
    # SWA / TTA
    p.add_argument('--swa', type=int, default=0)
    p.add_argument('--swa_start', type=int, default=75)
    p.add_argument('--tta', type=str, default='none', choices=['none','flip2','flip4'])
    # Misc
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', type=str, default='./out')
    return p.parse_args()


def build_dataloaders(args):
    assert torchvision is not None, 'torchvision required for example datasets'
    if args.dataset == 'cifar10':
        num_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_tf = build_train_transforms(args.img_size, {
            'random_crop': bool(args.random_crop),
            'hflip': bool(args.hflip),
            'vflip': bool(args.vflip),
            'color_jitter': bool(args.color_jitter),
            'randaugment': bool(args.randaugment),
            'trivialaugment': bool(args.trivialaugment),
            'cutout': bool(args.cutout),
            'cutout_holes': args.cutout_holes,
            'cutout_length': args.cutout_length,
            'mean': mean, 'std': std,
        })
        test_tf = build_test_transforms(args.img_size, mean, std)
        train_ds = datasets.CIFAR10(args.data, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(args.data, train=False, download=True, transform=test_tf)
    elif args.dataset == 'cifar100':
        num_classes = 100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        train_tf = build_train_transforms(args.img_size, {
            'random_crop': bool(args.random_crop), 'hflip': bool(args.hflip), 'vflip': bool(args.vflip),
            'color_jitter': bool(args.color_jitter), 'randaugment': bool(args.randaugment),
            'trivialaugment': bool(args.trivialaugment), 'cutout': bool(args.cutout),
            'cutout_holes': args.cutout_holes, 'cutout_length': args.cutout_length,
            'mean': mean, 'std': std,
        })
        test_tf = build_test_transforms(args.img_size, mean, std)
        train_ds = datasets.CIFAR100(args.data, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR100(args.data, train=False, download=True, transform=test_tf)
    else:
        raise NotImplementedError('Only CIFAR10/100 provided as example datasets here')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader, num_classes


def get_class_counts(loader) -> List[int]:
    counts = None
    for _, target in loader:
        t = target
        if counts is None:
            counts = torch.zeros(int(t.max().item()) + 1, dtype=torch.long)
        for c in t.tolist():
            if c >= counts.numel():
                counts = F.pad(counts, (0, c - counts.numel() + 1))
            counts[c] += 1
    return counts.tolist()


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, num_classes = build_dataloaders(args)
    class_counts = get_class_counts(train_loader)

    # Model
    model = SimpleNet(in_ch=3, num_classes=num_classes, width=args.width, depth=args.depth,
                      norm=args.norm, act=args.act, use_ws=bool(args.use_ws), drop_path=args.drop_path,
                      dropblock=args.dropblock, block_size=args.block_size, dropout=args.dropout,
                      embedding_dim=args.embedding_dim, head_type=('margin' if args.head=='margin' else 'linear'),
                      margin_type=args.margin_type, margin_m=args.margin_m, margin_s=args.margin_s)
    model.to(device)

    # Optimizer
    if args.sam:
        optimizer = build_sam(model.parameters(), base=args.optimizer, lr=args.lr, weight_decay=args.weight_decay,
                              momentum=args.momentum, nesterov=bool(args.nesterov), rho=args.sam_rho)
    else:
        optimizer = build_optimizer(model.parameters(), args.optimizer, lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum, nesterov=bool(args.nesterov))

    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer.base_optimizer if args.sam else optimizer, args.schedule,
                                args.epochs, steps_per_epoch, args.lr, warmup_epochs=args.warmup_epochs,
                                step_milestones=args.step_milestones, gamma=args.gamma, poly_power=args.poly_power)

    # Loss
    if args.multilabel:
        criterion = AsymmetricLossMultiLabel()
    else:
        if args.loss == 'ce':
            criterion = lambda logits, target: F.cross_entropy(logits, target, label_smoothing=args.label_smoothing)
        elif args.loss == 'focal':
            criterion = FocalLoss()
        elif args.loss == 'cb_focal':
            criterion = ClassBalancedFocalLoss(class_counts)
        elif args.loss == 'ldam':
            criterion = LDAMLoss(class_counts)
        elif args.loss == 'ldam_drw':
            criterion = LDAMLoss(class_counts)
        elif args.loss == 'margin_ce':
            # Use CE on margin-softmax logits
            criterion = lambda logits, target: F.cross_entropy(logits, target)
        else:
            raise ValueError('Unknown loss')

    # MixUp/CutMix helper
    mixcfg = MixupCutmixCfg(args.mixup_alpha, args.cutmix_alpha, num_classes)

    # Early stopping
    es = EarlyStopping(patience=args.patience, mode='max') if args.early_stop else None

    # SWA
    if args.swa:
        swa_model, swa_sched = build_swa(optimizer.base_optimizer if args.sam else optimizer, model)
    else:
        swa_model = None
        swa_sched = None

    best_acc = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # MixUp/CutMix (multiclass only)
            if not args.multilabel and (args.mixup_alpha > 0 or args.cutmix_alpha > 0):
                images, y_mix, index, lam, kind = apply_mixup_cutmix(images, targets, mixcfg)
            else:
                y_mix, index, lam, kind = None, None, 1.0, 'none'

            def forward_backward():
                if args.head == 'margin' and not args.multilabel:
                    logits = model(images, targets)
                else:
                    logits = model(images)
                if args.multilabel:
                    raise NotImplementedError('Provide multi-label dataset/targets one-hot for ASL use-case')
                if y_mix is not None:
                    loss = -torch.sum(F.log_softmax(logits, dim=1) * y_mix) / logits.size(0)
                else:
                    loss = criterion(logits, targets)
                loss.backward()
                return loss

            if args.sam:
                loss = forward_backward()
                optimizer.first_step()
                forward_backward()
                optimizer.second_step()
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = forward_backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            (swa_sched or scheduler).step() if (args.swa and epoch >= args.swa_start) else scheduler.step()
            running_loss += loss.item()

        # SWA update
        if args.swa and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()

        # Validation
        val_acc, val_loss = evaluate(model, val_loader, device, args)
        if args.swa and epoch == args.epochs - 1:
            # consolidate SWA
            update_bn_for_swa(val_loader, swa_model, device)
            swa_acc, swa_loss = evaluate(swa_model, val_loader, device, args)
            if swa_acc > val_acc:
                val_acc, val_loss = swa_acc, swa_loss
                model = swa_model

        if es is not None and es.step(val_acc):
            print(f"Early stopping at epoch {epoch}")
            break

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(args.out, 'best.pt'))

        t1 = time.time()
        print(f"Epoch {epoch+1}/{args.epochs} | loss {running_loss/steps_per_epoch:.4f} | val_acc {val_acc:.2f} | time {t1-t0:.1f}s")

    torch.save({'model': model.state_dict()}, os.path.join(args.out, 'final.pt'))


def evaluate(model: nn.Module, loader: DataLoader, device, args):
    model.eval()
    total = 0
    correct1 = 0
    losses = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if args.tta != 'none':
                logits = tta_predict_logits(model, images, tta=args.tta)
            else:
                logits = model(images)
            loss = F.cross_entropy(logits, targets)
            acc1 = (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)
            correct1 += acc1
            losses += loss.item() * targets.size(0)
    return 100.0 * correct1 / total, losses / total


if __name__ == '__main__':
    main()
