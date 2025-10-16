"""
Building blocks for supervised single-model training with:
- Losses: CE (with label smoothing), Focal, Class-Balanced Focal, LDAM (+DRW), Margin-Softmax (ArcFace/CosFace/AM-Softmax), Asymmetric Loss (multi-label)
- Data augmentation: crop/flip/jitter, Cutout, MixUp, CutMix, RandAugment, TrivialAugmentWide (fixed policies)
- Regularization: weight decay, dropout, DropBlock, Stochastic Depth/DropPath, ShakeDrop, gradient clipping (in train loop), early stopping util
- Optimizers & schedulers: SGD+Nesterov, AdamW, cosine/SGDR, OneCycleLR, warmup+step/poly, SAM
- Norm & activations: BatchNorm, GroupNorm, (Weight Standardization), EvoNorm (S0, B0), ReLU/LeakyReLU/PReLU, ELU/SELU, GELU, SiLU/Swish, Mish
- Inference-time: TTA (flip/resize), SWA helpers

PyTorch ≥ 1.12 recommended (works on 2.x). torchvision ≥ 0.13 for RandAugment/TrivialAugmentWide.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

try:
    import torchvision
    from torchvision import transforms
except Exception:
    torchvision = None
    transforms = None

# -------------------- Utilities -------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# -------------------- Activations -------------------- #

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def act_factory(name: str) -> nn.Module:
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leakyrelu':
        return nn.LeakyReLU(0.01, inplace=True)
    if name == 'prelu':
        return nn.PReLU()
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'selu':
        return nn.SELU(inplace=True)
    if name == 'gelu':
        return nn.GELU()
    if name in ['silu', 'swish']:
        return nn.SiLU(inplace=True)
    if name == 'mish':
        return Mish()
    raise ValueError(f"Unknown activation: {name}")

# -------------------- Normalizations & WS -------------------- #

class EvoNormS0(nn.Module):
    """EvoNorm-S0 (sample-based). Reference: Liu 2020.
    This variant does not use batch statistics; good for small batches.
    """
    def __init__(self, channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1)) if affine else None
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1)) if affine else None
        self.v = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        # instance-like variance per sample
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        den = torch.sqrt(var + self.eps)
        nx = x / den
        if self.gamma is not None:
            nx = nx * self.gamma
        nx = nx + self.beta if self.beta is not None else nx
        # non-linear term
        return nx * torch.sigmoid(self.v * x)


class EvoNormB0(nn.Module):
    """EvoNorm-B0 (batch-based)."""
    def __init__(self, channels: int, eps: float = 1e-5, affine: bool = True, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1)) if affine else None
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1)) if affine else None
        self.register_buffer('running_var', torch.ones(1, channels, 1, 1))

    def forward(self, x):
        if self.training:
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            var = self.running_var
        den = torch.sqrt(var + self.eps)
        nx = x / den
        if self.gamma is not None:
            nx = nx * self.gamma
        nx = nx + self.beta if self.beta is not None else nx
        return nx


class Conv2dWS(nn.Conv2d):
    """Conv2d with Weight Standardization.
    Use with GroupNorm/BatchNorm for WS+Norm stacks.
    """
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight = weight - weight_mean
        std = weight.flatten(1).std(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def norm_factory(name: str, num_channels: int, groups: int = 32) -> nn.Module:
    name = name.lower()
    if name == 'batchnorm':
        return nn.BatchNorm2d(num_channels)
    if name == 'groupnorm':
        g = min(groups, num_channels)
        return nn.GroupNorm(g, num_channels)
    if name == 'evonorm_s0':
        return EvoNormS0(num_channels)
    if name == 'evonorm_b0':
        return EvoNormB0(num_channels)
    if name == 'identity' or name == 'none':
        return nn.Identity()
    raise ValueError(f"Unknown norm: {name}")

# -------------------- Stochastic Regularizers -------------------- #

class DropPath(nn.Module):
    """Stochastic Depth/DropPath.
    Drop paths (residual branches) per sample (when applied in main path, acts as dropout on features).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * random_tensor


class DropBlock2D(nn.Module):
    """DropBlock for 2D feature maps.
    Reference: Ghiasi 2018. Works best with increasing drop_prob during training.
    """
    def __init__(self, block_size: int = 7, drop_prob: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        gamma = self._compute_gamma(x)
        mask = (torch.rand(x.shape[0], *x.shape[2:], device=x.device) < gamma).float()
        mask = F.max_pool2d(mask.unsqueeze(1), kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask
        out = x * mask
        # rescale
        out = out * mask.numel() / mask.sum().clamp(min=1.0)
        return out

    def _compute_gamma(self, x):
        _, _, h, w = x.shape
        return self.drop_prob * (h * w) / ((self.block_size ** 2) * (h - self.block_size + 1) * (w - self.block_size + 1))


class ShakeDrop(nn.Module):
    """ShakeDrop regularizer (single-network variant).
    Insert in residual branches. During training, randomly scales residual with random factors in forward/backward.
    Reference: Yamada 2018 (ShakeDrop).
    """
    def __init__(self, p_drop: float = 0.0, alpha_range: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        if not self.training:
            return x
        if torch.rand(1).item() < self.p_drop:
            alpha = torch.empty(1).uniform_(*self.alpha_range).to(x.device)
            beta = torch.empty(1).uniform_(*self.alpha_range).to(x.device)
            y = alpha * x + (1 - alpha) * x.detach()  # straight-through like
            # backward uses beta implicitly via detach difference
            return y + (beta - alpha) * x.detach()
        return x

# -------------------- Simple Example Backbone -------------------- #

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm: str = 'batchnorm', act: str = 'relu', drop_path: float = 0.0,
                 dropblock: float = 0.0, block_size: int = 7, use_ws: bool = False):
        super().__init__()
        Conv = Conv2dWS if use_ws else nn.Conv2d
        self.conv = Conv(in_ch, out_ch, 3, padding=1, bias=(norm in ['identity', 'none']))
        self.norm = norm_factory(norm, out_ch)
        self.act = act_factory(act)
        self.db = DropBlock2D(block_size, dropblock) if dropblock > 0 else nn.Identity()
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.db(y)
        return self.dp(y)


class SimpleNet(nn.Module):
    """A compact CNN to demonstrate blocks. Replace with your own backbone if desired.
    Supports optional WS convs, EvoNorm/BN/GN, DropBlock, DropPath, and pluggable activations.
    """
    def __init__(self, in_ch=3, num_classes=10, width=64, depth=4, norm='batchnorm', act='relu',
                 use_ws=False, drop_path=0.0, dropblock=0.0, block_size=7, dropout=0.0,
                 embedding_dim: int = 256, head_type: str = 'linear', margin_type: str = 'arcface',
                 margin_m: float = 0.5, margin_s: float = 30.0):
        super().__init__()
        channels = [width * (2 ** i) for i in range(depth)]
        blocks = []
        c_in = in_ch
        for i, c in enumerate(channels):
            blocks.append(ConvBlock(c_in, c, norm, act, drop_path * (i + 1) / depth, dropblock, block_size, use_ws))
            blocks.append(nn.Conv2d(c, c, 3, stride=2, padding=1))  # downsample
            blocks.append(norm_factory(norm, c))
            blocks.append(act_factory(act))
            c_in = c
        self.stem = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(channels[-1], embedding_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head_type = head_type.lower()
        if self.head_type == 'linear':
            self.head = nn.Linear(embedding_dim, num_classes)
        elif self.head_type == 'margin':
            self.head = MarginSoftmaxHead(embedding_dim, num_classes, typ=margin_type, m=margin_m, s=margin_s)
        else:
            raise ValueError("head_type must be 'linear' or 'margin'")

    def forward_features(self, x):
        x = self.stem(x)
        x = self.pool(x).flatten(1)
        x = self.embed(x)
        x = self.dropout(x)
        return F.normalize(x) if isinstance(self.head, MarginSoftmaxHead) else x

    def forward(self, x, y: Optional[torch.Tensor] = None):
        z = self.forward_features(x)
        if isinstance(self.head, MarginSoftmaxHead):
            if self.training and y is None:
                raise ValueError('Margin head requires labels y during training')
            logits = self.head(z, y)
        else:
            logits = self.head(z)
        return logits

# -------------------- Margin-Softmax Heads -------------------- #

class MarginSoftmaxHead(nn.Module):
    def __init__(self, in_features, num_classes, typ='arcface', m=0.5, s=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.m = m
        self.s = s
        self.typ = typ.lower()

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # normalize
        W = F.normalize(self.weight)
        x = F.normalize(features)
        logits = F.linear(x, W)  # cosine similarity
        if self.training and labels is not None:
            theta = torch.acos(logits.clamp(-1 + 1e-7, 1 - 1e-7))
            if self.typ == 'arcface':
                target_logit = torch.cos(theta + self.m)
            elif self.typ == 'cosface':
                target_logit = logits - self.m
            elif self.typ in ['am-softmax', 'amsoftmax', 'am']:
                target_logit = logits - self.m
            else:
                raise ValueError('Unknown margin softmax type')
            one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
            logits = logits * (1 - one_hot) + target_logit * one_hot
        return logits * self.s

# -------------------- Losses -------------------- #

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            at = torch.full_like(target, self.alpha, dtype=torch.float)
            at = at.to(loss.device)
            loss = at * loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class ClassBalancedFocalLoss(nn.Module):
    """Uses effective number of samples as per Cui 2019 to reweight focal loss."""
    def __init__(self, class_counts: List[int], beta: float = 0.9999, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        cls = torch.tensor(class_counts, dtype=torch.float)
        effective_num = 1.0 - torch.pow(beta, cls)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer('weights', weights)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        w = self.weights[target]
        loss = w * focal
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class LDAMLoss(nn.Module):
    """LDAM loss with optional class reweighting (use with DRW schedule)."""
    def __init__(self, class_counts: List[int], max_m: float = 0.5, s: float = 30.0):
        super().__init__()
        cls = torch.tensor(class_counts, dtype=torch.float)
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer('m_list', m_list)
        self.s = s

    def forward(self, logits, target):
        index = F.one_hot(target, num_classes=logits.size(1)).float()
        batch_m = torch.matmul(self.m_list[None, :], index.t()).view(-1, 1)
        logits_m = logits - index * batch_m
        return F.cross_entropy(self.s * logits_m, target)


def drw_class_weights(epoch: int, epochs: int, class_counts: List[int], beta_high=0.9999, beta_low=0.9, milestone_ratio: float = 0.5) -> torch.Tensor:
    milestone = int(epochs * milestone_ratio)
    beta = beta_high if epoch >= milestone else beta_low
    cls = torch.tensor(class_counts, dtype=torch.float)
    effective_num = 1.0 - torch.pow(beta, cls)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(class_counts)
    return weights


class AsymmetricLossMultiLabel(nn.Module):
    """ASL for multi-label classification.
    Reference: Ridnik 2021.
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits, target):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        loss = target * torch.log(xs_pos.clamp(min=self.eps)) + (1 - target) * torch.log(xs_neg.clamp(min=self.eps))
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)
        pt0 = xs_pos * target
        pt1 = xs_neg * (1 - target)
        one_sided_gamma = self.gamma_pos * target + self.gamma_neg * (1 - target)
        one_sided_w = torch.pow(1 - pt0 - pt1, one_sided_gamma)
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)
        loss *= one_sided_w
        return -loss.mean()

# -------------------- Augmentations -------------------- #

class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        if not torch.is_tensor(img):
            img = transforms.ToTensor()(img)
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        for _ in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        mask = mask.expand_as(img)
        return img * mask


def build_train_transforms(img_size: int = 224, aug_cfg: dict | None = None):
    if transforms is None:
        raise RuntimeError('torchvision is required for augmentations')
    aug_cfg = aug_cfg or {}
    ops = []
    # Basic spatial/color
    if aug_cfg.get('random_crop', True):
        ops.append(transforms.RandomResizedCrop(img_size))
    else:
        ops.append(transforms.Resize((img_size, img_size)))
    if aug_cfg.get('hflip', True):
        ops.append(transforms.RandomHorizontalFlip())
    if aug_cfg.get('vflip', False):
        ops.append(transforms.RandomVerticalFlip())
    if aug_cfg.get('color_jitter', False):
        ops.append(transforms.ColorJitter(*aug_cfg.get('cj_params', (0.4, 0.4, 0.4, 0.1))))
    # Advanced policy-based (fixed policies, no controller)
    if aug_cfg.get('randaugment', False):
        ops.append(transforms.RandAugment())
    if aug_cfg.get('trivialaugment', False):
        ops.append(transforms.TrivialAugmentWide())
    ops.append(transforms.ToTensor())
    # Cutout after ToTensor
    if aug_cfg.get('cutout', False):
        ops.append(Cutout(aug_cfg.get('cutout_holes', 1), aug_cfg.get('cutout_length', 16)))
    # Normalize if provided
    if 'mean' in aug_cfg and 'std' in aug_cfg:
        ops.append(transforms.Normalize(mean=aug_cfg['mean'], std=aug_cfg['std']))
    return transforms.Compose(ops)


def build_test_transforms(img_size: int = 224, mean=None, std=None):
    if transforms is None:
        raise RuntimeError('torchvision is required for augmentations')
    ops = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    if mean is not None and std is not None:
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


@dataclass
class MixupCutmixCfg:
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    num_classes: int = 1000


def mixup_targets(target: torch.Tensor, num_classes: int, lam: float):
    y1 = F.one_hot(target, num_classes).float()
    index = torch.randperm(target.size(0)).to(target.device)
    y2 = F.one_hot(target[index], num_classes).float()
    return y1 * lam + y2 * (1 - lam), index


def apply_mixup_cutmix(inputs, targets, cfg: MixupCutmixCfg):
    lam = 1.0
    index = None
    if cfg.mixup_alpha > 0 and cfg.cutmix_alpha > 0:
        if random.random() < 0.5:
            lam = torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample().item()
            y, index = mixup_targets(targets, cfg.num_classes, lam)
            inputs = lam * inputs + (1 - lam) * inputs[torch.randperm(inputs.size(0)).to(inputs.device)]
            return inputs, y, index, lam, 'mixup'
        else:
            lam = torch.distributions.Beta(cfg.cutmix_alpha, cfg.cutmix_alpha).sample().item()
            index = torch.randperm(inputs.size(0)).to(inputs.device)
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
            y1 = F.one_hot(targets, cfg.num_classes).float()
            y2 = F.one_hot(targets[index], cfg.num_classes).float()
            y = y1 * lam + y2 * (1 - lam)
            return inputs, y, index, lam, 'cutmix'
    elif cfg.mixup_alpha > 0:
        lam = torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample().item()
        y, index = mixup_targets(targets, cfg.num_classes, lam)
        inputs = lam * inputs + (1 - lam) * inputs[torch.randperm(inputs.size(0)).to(inputs.device)]
        return inputs, y, index, lam, 'mixup'
    elif cfg.cutmix_alpha > 0:
        lam = torch.distributions.Beta(cfg.cutmix_alpha, cfg.cutmix_alpha).sample().item()
        index = torch.randperm(inputs.size(0)).to(inputs.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
        y1 = F.one_hot(targets, cfg.num_classes).float()
        y2 = F.one_hot(targets[index], cfg.num_classes).float()
        y = y1 * lam + y2 * (1 - lam)
        return inputs, y, index, lam, 'cutmix'
    else:
        return inputs, targets, None, 1.0, 'none'


def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    bbx1 = np_clip(cx - cut_w // 2, 0, W)
    bby1 = np_clip(cy - cut_h // 2, 0, H)
    bbx2 = np_clip(cx + cut_w // 2, 0, W)
    bby2 = np_clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def np_clip(val, low, high):
    return int(max(low, min(high, val)))

# -------------------- Optimizers / SAM / Schedules -------------------- #

class SAM(Optimizer):
    """Sharpness-Aware Minimization (single-model)."""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.rho = rho
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                p.state['e_w'] = e_w
        self.base_optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in p.state:
                    p.add_(-p.state['e_w'])
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()

    def step(self):
        raise RuntimeError('Call first_step and second_step for SAM')

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        norm = torch.norm(torch.stack([
            p.grad.norm(p=2) for group in self.param_groups for p in group['params'] if p.grad is not None
        ]), p=2)
        return norm


def build_optimizer(params, name: str, lr: float, weight_decay: float = 0.0, momentum: float = 0.9, nesterov: bool = True):
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError('Unknown optimizer')


def build_sam(params, base: str, lr: float, weight_decay: float = 0.0, momentum: float = 0.9, nesterov: bool = True, rho: float = 0.05):
    base_opt = lambda p, **kw: build_optimizer(p, base, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    return SAM(params, base_opt, rho=rho)


def build_scheduler(optimizer: Optimizer, schedule: str, epochs: int, steps_per_epoch: int, base_lr: float,
                    warmup_epochs: int = 0, step_milestones: Optional[List[int]] = None, gamma: float = 0.1,
                    t_max: Optional[int] = None, T_0: int = 10, T_mult: int = 2, poly_power: float = 2.0):
    schedule = schedule.lower()
    if schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
    if schedule == 'sgdr':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0 * steps_per_epoch, T_mult=T_mult)
    if schedule == 'onecycle':
        total_steps = epochs * steps_per_epoch
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr, total_steps=total_steps)
    if schedule == 'warmup_step':
        def lr_lambda(step):
            epoch = step / steps_per_epoch
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            # after warmup, step decay
            milestones = step_milestones or [int(0.6 * epochs), int(0.85 * epochs)]
            factor = 1.0
            for m in milestones:
                if epoch >= m:
                    factor *= gamma
            return factor
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if schedule == 'warmup_poly':
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return (1 - t) ** poly_power
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError('Unknown schedule')

# -------------------- Early Stopping -------------------- #

class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = 'min', min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0
        self.should_stop = False

    def step(self, metric: float):
        if self.best is None:
            self.best = metric
            return False
        improve = (metric < self.best - self.min_delta) if self.mode == 'min' else (metric > self.best + self.min_delta)
        if improve:
            self.best = metric
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return self.should_stop

# -------------------- TTA & SWA Helpers -------------------- #

@torch.no_grad()
def tta_predict_logits(model: nn.Module, x: torch.Tensor, tta: str = 'flip4'):
    model.eval()
    if tta == 'none':
        return model(x)
    logits = []
    if tta in ['flip2', 'flip4']:
        logits.append(model(x))
        logits.append(model(torch.flip(x, dims=[-1])))  # H-flip
        if tta == 'flip4':
            logits.append(model(torch.flip(x, dims=[-2])))  # V-flip
            logits.append(model(torch.flip(x, dims=[-1, -2])))
    else:
        logits.append(model(x))
    return torch.stack(logits, dim=0).mean(0)


def build_swa(optimizer: Optimizer, model: nn.Module):
    from torch.optim.swa_utils import AveragedModel, SWALR
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=optimizer.param_groups[0]['lr'])
    return swa_model, swa_scheduler


def update_bn_for_swa(loader, swa_model: nn.Module, device):
    from torch.optim.swa_utils import update_bn
    swa_model.train()
    update_bn(loader, swa_model, device=device)

# -------------------- END -------------------- #
