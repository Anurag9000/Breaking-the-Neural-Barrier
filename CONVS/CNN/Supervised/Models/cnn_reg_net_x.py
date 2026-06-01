"""
RegNetX (CIFAR) — single-model supervised.

Implements RegNetX family (Radosavovic et al., 2020) with block-wise widths generated from
(w0, wa, wm, d) and grouped 3×3 convs using group width gw. CIFAR tweaks: 3×3 stem (s=1),
then 4 stages with strides [1, 2, 2, 2] → spatial sizes 32→16→8→4. No SE (X variant).

Block: BottleneckX (bn-relu-1×1 → bn-relu-3×3 group conv → bn-1×1, residual when shape matches).
Bottleneck ratio = 1.0 (per RegNetX). Head: GAP → FC.

Includes common presets: X-200MF, X-400MF, X-600MF, X-800MF, X-1.6GF.
Factory mirrors your style and includes param_count helper.
"""
from __future__ import annotations
from typing import List, Tuple
import math
import torch
import torch.nn as nn

__all__ = [
    "RegNetXCIFAR",
    "make_regnetx_cifar",
]

# ----------------------
# Utils: RegNet width/depth generation
# ----------------------

def generate_regnet(w0: float, wa: float, wm: float, d: int, q: int = 8) -> Tuple[List[int], List[int]]:
    """Generate per-block widths (quantized) and per-stage summary.
    Returns (widths, block_counts_per_stage) after collapsing consecutive equal widths.
    """
    assert w0 > 0 and wa > 0 and wm > 1 and d > 0
    widths_cont = [w0 + wa * i for i in range(d)]
    widths = [int(round(wm ** round(math.log(w / w0) / math.log(wm)) * w0)) for w in widths_cont]
    widths = [int(round(w / q) * q) for w in widths]
    # collapse to stages
    stages: List[int] = []
    stage_ws: List[int] = []
    prev = None
    for w in widths:
        if w != prev:
            stage_ws.append(w)
            stages.append(1)
            prev = w
        else:
            stages[-1] += 1
    return stage_ws, stages

# Common RegNetX presets (ImageNet) — reused for CIFAR backbone widths
PRESETS = {
    "regnetx_200mf": dict(w0=24.0,  wa=36.44, wm=2.49, d=13, gw=8),
    "regnetx_400mf": dict(w0=24.0,  wa=24.48, wm=2.54, d=22, gw=16),
    "regnetx_600mf": dict(w0=48.0,  wa=24.48, wm=2.54, d=16, gw=24),
    "regnetx_800mf": dict(w0=56.0,  wa=24.48, wm=2.54, d=16, gw=16),
    "regnetx_1.6gf": dict(w0=80.0,  wa=34.56, wm=2.28, d=18, gw=24),
}

# ----------------------
# Blocks
# ----------------------
class BNAct(nn.Sequential):
    def __init__(self, ch: int, act: bool = True):
        layers = [nn.BatchNorm2d(ch)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)

class BottleneckX(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, group_w: int):
        super().__init__()
        assert stride in (1, 2)
        bottleneck_ratio = 1.0
        mid = int(round(out_ch * bottleneck_ratio))
        groups = max(1, mid // group_w)
        # 1×1 reduce
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn1 = BNAct(mid, act=True)
        # 3×3 group conv
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = BNAct(mid, act=True)
        # 1×1 expand
        self.conv3 = nn.Conv2d(mid, out_ch, kernel_size=1, bias=False)
        self.bn3 = BNAct(out_ch, act=False)
        # Shortcut
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.down = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x); out = self.bn1(out)
        out = self.conv2(out); out = self.bn2(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.down is not None:
            identity = self.down(x)
        out = self.relu(out + identity)
        return out

# ----------------------
# Network
# ----------------------
class RegNetXCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, preset: str = "regnetx_400mf"):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {list(PRESETS.keys())}")
        cfg = PRESETS[preset]
        stage_ws, stage_ds = generate_regnet(cfg['w0'], cfg['wa'], cfg['wm'], cfg['d'])
        # Align stage widths to be multiple of group width
        gw = cfg['gw']
        stage_ws = [max(gw, int(round(w / gw) * gw)) for w in stage_ws]

        # Stem (CIFAR: stride 1)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        in_ch = 32

        # Build 4 stages; if generated stages > 4, merge into 4 by splitting counts
        # For simplicity, map into 4 blocks by distributing depths
        if len(stage_ws) < 4:
            # pad by repeating last
            while len(stage_ws) < 4:
                stage_ws.append(stage_ws[-1])
                stage_ds.append(1)
        elif len(stage_ws) > 4:
            # merge tail stages into 4 bins
            bins = 4
            chunk = math.ceil(len(stage_ws) / bins)
            new_ws, new_ds = [], []
            for i in range(0, len(stage_ws), chunk):
                new_ws.append(stage_ws[i])
                new_ds.append(sum(stage_ds[i:i+chunk]))
            stage_ws, stage_ds = new_ws[:4], new_ds[:4]
        # Stage strides [1,2,2,2]
        strides = [1,2,2,2]
        stages: List[nn.Module] = []
        for w, d, s in zip(stage_ws[:4], stage_ds[:4], strides):
            stages.append(self._make_stage(in_ch, w, depth=d, stride=s, group_w=gw))
            in_ch = w
        self.stage1, self.stage2, self.stage3, self.stage4 = stages

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_ch, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def _make_stage(self, in_ch: int, out_ch: int, depth: int, stride: int, group_w: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.append(BottleneckX(in_ch, out_ch, stride=stride, group_w=group_w))
        for _ in range(depth - 1):
            layers.append(BottleneckX(out_ch, out_ch, stride=1, group_w=group_w))
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

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_regnetx_cifar(num_classes: int = 10, in_channels: int = 3, preset: str = "regnetx_400mf") -> RegNetXCIFAR:
    return RegNetXCIFAR(num_classes=num_classes, in_channels=in_channels, preset=preset)
