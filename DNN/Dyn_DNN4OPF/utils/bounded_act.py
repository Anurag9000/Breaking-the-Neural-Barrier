import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

class BoundedAct(nn.Module):
    """
    Element-wise clamp layer with optional on/off mask.

    • bounds_low  –  (out_dim,) tensor of lower limits
    • bounds_high –  (out_dim,) tensor of upper limits
    • mask        –  (out_dim,) {0,1} ints; 1 ⇒ activate clamp, 0 ⇒ pass through
    """
    def __init__(self,
                 bounds_low: torch.Tensor,
                 bounds_high: torch.Tensor,
                 mask: torch.Tensor):
        super().__init__()

        # ensure tensors
        bounds_low = torch.tensor(bounds_low, dtype=torch.float32)   \
                         if not isinstance(bounds_low, torch.Tensor) else bounds_low
        bounds_high = torch.tensor(bounds_high, dtype=torch.float32) \
                          if not isinstance(bounds_high, torch.Tensor) else bounds_high
        mask = torch.tensor(mask, dtype=torch.bool)                   \
                   if not isinstance(mask, torch.Tensor) else mask.bool()

        # register the bounds & mask
        self.register_buffer("lo",   bounds_low.reshape(1, -1))
        self.register_buffer("hi",   bounds_high.reshape(1, -1))
        self.register_buffer("mask", mask.reshape(1, -1))

        # ── run-time toggle for clipping ───────────────────────────────────
        #   • Training / validation → apply_bounds = False
        #   • Held-out test         → apply_bounds = True
        # The flag is a buffer so it moves with the model (to device, DDP, etc.).
        self.register_buffer("apply_bounds", torch.tensor(True, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fast path – clipping disabled or nothing to clamp
        if (not self.apply_bounds.item()) or (not self.mask.any()):
            return x
        # perform element-wise clamp
        clamp_part = torch.clamp(x, self.lo, self.hi)
        return torch.where(self.mask, clamp_part, x)
