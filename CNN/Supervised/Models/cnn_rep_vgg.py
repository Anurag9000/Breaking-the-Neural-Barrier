"""
RepVGG (CIFAR) — single-model supervised.

Implements RepVGG (Ding et al., 2021): VGG-style plain 3×3 conv blocks at inference time,
with training-time multi-branch re-parameterization (3×3 conv-BN, 1×1 conv-BN, and identity-BN).
CIFAR tweaks: stem stride=1; stage downsampling at the first block of stages 2–4 → sizes 32→16→8→4.

Block: (3×3 conv-BN) + (1×1 conv-BN) + (identity BN if in==out & stride=1) → sum → ReLU.
At deploy, branches are fused into a single 3×3 conv (no BN) for speed.

Presets (depth per stage, base channels): A0, A1, B0 (common, compact for CIFAR). Factory mirrors your style
and includes param_count + a .reparameterize() method to fuse all blocks.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

__all__ = [
    "RepVGGCIFAR",
    "make_repvgg_cifar",
]

# ----------------------
# Utilities
# ----------------------

def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Conv2d and BatchNorm2d into equivalent conv weight & bias (returns tensors)."""
    if conv is None:
        # handle identity branch turned into conv later
        raise ValueError("conv cannot be None in fuse_conv_bn")
    w = conv.weight
    if conv.bias is None:
        bias = torch.zeros(w.size(0), device=w.device)
    else:
        bias = conv.bias
    bn_var_rsqrt = (bn.running_var + bn.eps).rsqrt()
    scale = bn.weight * bn_var_rsqrt
    w_fused = w * scale.view(-1, 1, 1, 1)
    b_fused = (bias - bn.running_mean) * bn_var_rsqrt * bn.weight + bn.bias
    return w_fused, b_fused


def pad_1x1_to_3x3_tensor(k: torch.Tensor) -> torch.Tensor:
    if k is None:
        return None
    if k.size(2) == 3:
        return k
    assert k.size(2) == 1
    out_ch, in_ch = k.size(0), k.size(1)
    k3 = torch.zeros((out_ch, in_ch, 3, 3), device=k.device, dtype=k.dtype)
    k3[:, :, 1:2, 1:2] = k
    return k3


def get_identity_kernel_3x3(ch: int) -> torch.Tensor:
    k = torch.zeros((ch, ch, 3, 3))
    for i in range(ch):
        k[i, i, 1, 1] = 1.0
    return k

# ----------------------
# RepVGG Block
# ----------------------
class RepVGGBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, deploy: bool = False):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.deploy = deploy
        padding = 1
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, bias=True)
        else:
            self.rbr_dense_3x3 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.rbr_dense_1x1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if out_ch == in_ch and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(out_ch)
            else:
                self.rbr_identity = None
        self.nonlinearity = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.nonlinearity(self.rbr_reparam(x))
        out = self.rbr_dense_3x3(x) + self.rbr_dense_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.nonlinearity(out)

    def _get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 3x3 branch
        k3, b3 = fuse_conv_bn(self.rbr_dense_3x3[0], self.rbr_dense_3x3[1])
        # 1x1 branch padded to 3x3
        k1, b1 = fuse_conv_bn(self.rbr_dense_1x1[0], self.rbr_dense_1x1[1])
        k1 = pad_1x1_to_3x3_tensor(k1)
        # identity branch (BN only) to conv3x3 equivalent
        if self.rbr_identity is not None:
            id_k = get_identity_kernel_3x3(self.out_ch).to(k3.device, k3.dtype)
            bn = self.rbr_identity
            bn_var_rsqrt = (bn.running_var + bn.eps).rsqrt()
            scale = bn.weight * bn_var_rsqrt
            k_id = id_k * scale.view(-1, 1, 1, 1)
            b_id = (-bn.running_mean) * bn_var_rsqrt * bn.weight + bn.bias
        else:
            k_id = torch.zeros_like(k3)
            b_id = torch.zeros_like(b3)
        # sum
        k = k3 + k1 + k_id
        b = b3 + b1 + b_id
        return k, b

    def reparameterize(self):
        """Fuse branches into a single conv for inference."""
        if self.deploy:
            return
        k, b = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, stride=self.stride, padding=1, bias=True)
        self.rbr_reparam.weight.data = k
        self.rbr_reparam.bias.data = b
        # delete training branches
        del self.rbr_dense_3x3
        del self.rbr_dense_1x1
        if self.rbr_identity is not None:
            del self.rbr_identity
        self.deploy = True

# ----------------------
# Network
# ----------------------
PRESETS = {
    # depths per stage [s1,s2,s3,s4], base channels
    "A0": dict(depths=[2, 4, 14, 1], base=32),
    "A1": dict(depths=[2, 6, 16, 2], base=32),
    "B0": dict(depths=[4, 6, 16, 2], base=48),
}

class RepVGGCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, preset: str = "A0", deploy: bool = False):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {list(PRESETS.keys())}")
        cfg = PRESETS[preset]
        depths = cfg["depths"]
        base = cfg["base"]
        # channel plan per stage like VGG doubling
        channels = [base, base*2, base*4, base*8]

        # Stem (stride=1)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        in_ch = base

        # Stages with downsampling at first block of stages 2–4
        self.stage1 = self._make_stage(in_ch, channels[0], depths[0], stride_first=1, deploy=deploy)
        in_ch = channels[0]
        self.stage2 = self._make_stage(in_ch, channels[1], depths[1], stride_first=2, deploy=deploy)
        in_ch = channels[1]
        self.stage3 = self._make_stage(in_ch, channels[2], depths[2], stride_first=2, deploy=deploy)
        in_ch = channels[2]
        self.stage4 = self._make_stage(in_ch, channels[3], depths[3], stride_first=2, deploy=deploy)
        in_ch = channels[3]

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_ch, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def _make_stage(self, in_ch: int, out_ch: int, depth: int, stride_first: int, deploy: bool) -> nn.Sequential:
        layers: List[nn.Module] = []
        # first block may downsample
        layers.append(RepVGGBlock(in_ch, out_ch, stride=stride_first, deploy=deploy))
        for _ in range(depth - 1):
            layers.append(RepVGGBlock(out_ch, out_ch, stride=1, deploy=deploy))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def reparameterize(self):
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.reparameterize()

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_repvgg_cifar(num_classes: int = 10, in_channels: int = 3, preset: str = "A0", deploy: bool = False) -> RepVGGCIFAR:
    return RepVGGCIFAR(num_classes=num_classes, in_channels=in_channels, preset=preset, deploy=deploy)
