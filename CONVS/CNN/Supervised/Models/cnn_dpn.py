"""
DPN (Dual Path Network) for CIFAR-10/100 — single-model supervised.

This is a CIFAR-friendly Dual Path Network that preserves the **dual-path connectivity** idea from
Chen et al., "Dual Path Networks" (2017): each block outputs
  • a **residual** part (summed with a projected identity), and
  • a **dense** part (concatenated with previous dense features),
so subsequent blocks receive a tensor that is conceptually [residual || dense].

Design choices for 32×32 inputs:
- Stem: 3×3 conv(64).
- 4 stages; the first block in stages 2–4 downsamples (stride=2).
- Grouped 3×3 convs (ResNeXt-style) with configurable `cardinality` and per-stage bottleneck width.
- Each block expands to [residual_out_channels + k] where `k` is the dense-growth for the stage.
- Head: BN→ReLU→GAP→Linear.

Factory exposes a compact configuration (default "CIFAR-compact": [2,2,2,2] units with modest widths) as well as a
heavier variant approximating DPN-92 depth pattern ("DPN92-lite": [3,4,20,3]) but channel counts kept reasonable for CIFAR.

Note: This implementation focuses on the canonical **algorithmic structure** of DPN in a single-model classifier
and matches your modular coding style. It does not use EMA/teacher/ensembles.
"""
from __future__ import annotations
from typing import List, Tuple
import math
import torch
import torch.nn as nn

__all__ = ["DPNCIFAR", "make_dpn_cifar", "DPNBlock"]

class DPNBlock(nn.Module):
    """Dual Path block working on a concatenated tensor [residual || dense].

    Args:
        in_res: channels belonging to residual path at input
        in_dense: channels belonging to dense path at input
        R: output residual channels for this block
        k: new dense channels this block will contribute (growth)
        groups: grouped conv groups for the 3×3 conv
        mid: bottleneck width before grouped conv (if None, set to R//2)
        stride: 1 or 2 (downsample)
    """
    def __init__(self, in_res: int, in_dense: int, R: int, k: int, groups: int = 16, mid: int | None = None, stride: int = 1):
        super().__init__()
        self.in_res = in_res
        self.in_dense = in_dense
        self.R = R
        self.k = k
        self.stride = stride

        total_in = in_res + in_dense
        mid = mid or max(R // 2, 32)
        # 1x1 reduce
        self.conv1 = nn.Conv2d(total_in, mid, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        # 3x3 grouped conv
        g = max(1, min(groups, mid))
        # ensure mid divisible by groups
        if mid % g != 0:
            g = math.gcd(mid, g)
            g = max(1, g)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)
        # 1x1 expand to residual + dense increment
        self.conv3 = nn.Conv2d(mid, R + k, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(R + k)
        self.relu  = nn.ReLU(inplace=True)

        # Projection for residual identity when shape changes
        self.proj = None
        if stride != 1 or in_res != R:
            self.proj = nn.Sequential(
                nn.Conv2d(in_res, R, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(R),
            )
        # Downsample op for dense stream when stride=2
        self.pool_dense = nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity()

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input into residual and dense parts
        xr, xd = x[:, :self.in_res], x[:, self.in_res:]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out_res = out[:, :self.R]
        out_den = out[:, self.R:]

        # Residual path: sum
        idr = xr if self.proj is None else self.proj(xr)
        yr = self.relu(idr + out_res)
        # Dense path: concat pooled prev dense with new features
        xd_ds = self.pool_dense(xd)
        yd = torch.cat([xd_ds, out_den], dim=1)
        # Merge back
        y = torch.cat([yr, yd], dim=1)
        return y

class DPNCIFAR(nn.Module):
    def __init__(self,
                 num_classes: int = 10,
                 in_channels: int = 3,
                 units: List[int] = [2,2,2,2],
                 R_list: List[int] = [64, 96, 128, 256],
                 k_list: List[int] = [16, 16, 32, 64],
                 groups: int = 16,
                 mid_list: List[int] | None = None):
        super().__init__()
        assert len(units) == 4 and len(R_list) == 4 and len(k_list) == 4
        if mid_list is None:
            mid_list = [max(R//2, 32) for R in R_list]

        # Stem
        self.stem = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Initial dual-path split: treat all 64 as residual part, zero dense
        in_res, in_den = 64, 0

        self.stages: nn.ModuleList = nn.ModuleList()
        self.out_channels = None
        strides = [1, 2, 2, 2]
        for stage_idx in range(4):
            blocks: List[nn.Module] = []
            R = R_list[stage_idx]
            k = k_list[stage_idx]
            mid = mid_list[stage_idx]
            n = units[stage_idx]
            for i in range(n):
                stride = strides[stage_idx] if i == 0 else 1
                block = DPNBlock(in_res, in_den, R=R, k=k, groups=groups, mid=mid, stride=stride)
                blocks.append(block)
                # Update in_res, in_den for next block
                in_res = R
                in_den = (in_den // (2 if stride == 2 else 1)) + k
            self.stages.append(nn.Sequential(*blocks))
        self.out_channels = in_res + in_den

        # Head
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.out_channels, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.stem_bn(self.stem(x)))
        for s in self.stages:
            x = s(x)
        x = self.relu(self.bn(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

# ---------------- Factory -------------------
def make_dpn_cifar(config: str = "compact", num_classes: int = 10, in_channels: int = 3,
                   groups: int = 16) -> DPNCIFAR:
    """Factory for CIFAR DPN.
    - "compact":   units=[2,2,2,2], R=[64,96,128,256], k=[16,16,32,64]
    - "dpn92lite": units=[3,4,20,3], R=[96,128,256,512], k=[16,16,32,128]
    """
    if config.lower() in ["compact", "cifar", "small"]:
        units = [2,2,2,2]
        R = [64, 96, 128, 256]
        k = [16, 16, 32, 64]
    elif config.lower() in ["dpn92lite", "92", "dpn92"]:
        units = [3,4,20,3]
        R = [96, 128, 256, 512]
        k = [16, 16, 32, 128]
    else:
        raise ValueError("Unknown DPN config. Use 'compact' or 'dpn92lite'.")
    return DPNCIFAR(num_classes=num_classes, in_channels=in_channels, units=units, R_list=R, k_list=k, groups=groups)
