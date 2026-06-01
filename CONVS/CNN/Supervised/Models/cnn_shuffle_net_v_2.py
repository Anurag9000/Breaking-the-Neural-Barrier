"""
ShuffleNet V2 (CIFAR) — single-model supervised.

Implements ShuffleNetV2 (Ma et al., 2018) with the standard **split-transform-merge** unit,
**channel shuffle**, and depthwise 3×3 convs. CIFAR-friendly: stem stride=1 (no early downsample),
then stride-2 units at the start of stages 2–4 → spatial sizes 32→16→8→4. GAP→FC head.

Width scales supported per paper presets: {0.5×, 1.0×, 1.5×, 2.0×}.
For 1.0× the stage channels are [24, 116, 232, 464] with final 1024.

Matches your modular + factory pattern, includes param_count helper.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

__all__ = [
    "ShuffleNetV2CIFAR",
    "make_shufflenet_v2_cifar",
]

# Preset output channels per width scale (stem=24, then s2, s3, s4, final)
_STAGE_OUT = {
    0.5:  [24,  48,  96, 192, 1024],
    1.0:  [24, 116, 232, 464, 1024],
    1.5:  [24, 176, 352, 704, 1024],
    2.0:  [24, 244, 488, 976, 2048],
}


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    N, C, H, W = x.size()
    assert C % groups == 0
    x = x.view(N, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(N, C, H, W)
    return x

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int | None = None, groups: int = 1, act: str = "relu"):
        if p is None:
            p = (k - 1) // 2
        act_layer = nn.ReLU(inplace=True) if act == "relu" else nn.Identity()
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            act_layer,
        )

class DWConvBN(nn.Sequential):
    def __init__(self, ch: int, k: int = 3, s: int = 1):
        super().__init__(
            nn.Conv2d(ch, ch, kernel_size=k, stride=s, padding=(k-1)//2, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
        )

class ShuffleUnitV2(nn.Module):
    """ShuffleNetV2 Unit
    - stride=1: split input into two branches [x1, x2]; branch1 identity; branch2: pw → dw → pw; concat → shuffle
    - stride=2: no split; branch1: dw(s=2) → pw; branch2: pw → dw(s=2) → pw; concat → shuffle
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        branch_out = out_ch // 2

        if stride == 1:
            # input channels must be even
            assert in_ch % 2 == 0
            self.branch1 = nn.Identity()
            self.branch2 = nn.Sequential(
                ConvBNAct(in_ch // 2, branch_out, k=1, s=1, p=0),
                DWConvBN(branch_out, k=3, s=1),
                ConvBNAct(branch_out, branch_out, k=1, s=1, p=0, act="none"),
            )
        else:
            # branch1 (no split)
            self.branch1 = nn.Sequential(
                DWConvBN(in_ch, k=3, s=2),
                ConvBNAct(in_ch, branch_out, k=1, s=1, p=0, act="none"),
            )
            # branch2
            self.branch2 = nn.Sequential(
                ConvBNAct(in_ch, branch_out, k=1, s=1, p=0),
                DWConvBN(branch_out, k=3, s=2),
                ConvBNAct(branch_out, branch_out, k=1, s=1, p=0, act="none"),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, groups=2)
        out = self.relu(out)
        return out

class ShuffleNetV2CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0):
        super().__init__()
        if width_mult not in _STAGE_OUT:
            raise ValueError(f"width_mult must be one of {list(_STAGE_OUT.keys())}")
        c_stem, c2, c3, c4, c_last = _STAGE_OUT[width_mult]

        # Stem (stride=1 for CIFAR)
        self.stem = ConvBNAct(in_channels, c_stem, k=3, s=1)

        # Stages: repeats per paper [4,8,4]
        repeats = [4, 8, 4]
        self.stage2 = self._make_stage(c_stem, c2, repeats[0], downsample=True)
        self.stage3 = self._make_stage(c2,    c3, repeats[1], downsample=True)
        self.stage4 = self._make_stage(c3,    c4, repeats[2], downsample=True)

        # Last 1×1 conv
        self.conv_last = ConvBNAct(c4, c_last, k=1, s=1, p=0, act="relu")
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(c_last, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def _make_stage(self, in_ch: int, out_ch: int, repeats: int, downsample: bool) -> nn.Sequential:
        layers: List[nn.Module] = []
        # first unit downsamples if requested
        layers.append(ShuffleUnitV2(in_ch, out_ch, stride=2 if downsample else 1))
        for _ in range(repeats - 1):
            layers.append(ShuffleUnitV2(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv_last(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_shufflenet_v2_cifar(num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0) -> ShuffleNetV2CIFAR:
    return ShuffleNetV2CIFAR(num_classes=num_classes, in_channels=in_channels, width_mult=width_mult)
