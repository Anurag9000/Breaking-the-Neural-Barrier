import random
from typing import Optional, Tuple

import torch
from torch import Tensor


def _rand_bbox(width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
    """Sample CutMix bounding box coordinates."""
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    # Uniform center position
    cx = random.randint(0, width - 1)
    cy = random.randint(0, height - 1)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def cutmix_batch(
    x: Tensor,
    y: Tensor,
    *,
    alpha: float = 1.0,
    p: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor, float]]]:
    """
    Apply CutMix to a batch with probability ``p``.

    Returns:
        mixed_x: potentially modified inputs.
        targets: ``None`` if CutMix not applied, otherwise ``(y1, y2, lam)``.
    """
    if alpha <= 0.0 or p <= 0.0 or x.size(0) <= 1:
        return x, None

    if random.random() > p:
        return x, None

    if generator is None:
        generator = torch.Generator(device=x.device)

    lam = torch.distributions.Beta(alpha, alpha).sample((1,)).item()
    batch_size, _, h, w = x.shape
    index = torch.randperm(batch_size, generator=generator, device=x.device)

    x1, y1, x2, y2 = _rand_bbox(w, h, lam)
    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda based on the actually mixed area
    area = (x2 - x1) * (y2 - y1)
    lam_adj = 1.0 - area / float(w * h)

    y1_t = y
    y2_t = y[index]
    return x_mixed, (y1_t, y2_t, float(lam_adj))

