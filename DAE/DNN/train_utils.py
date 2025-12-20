import time
from typing import Optional, Tuple

import torch
import torch.nn as nn


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            return batch[0], batch[1], None
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
    return batch, None, None


def train_epoch(model, loader, loss_fn, optimizer, device, task_type: str, grad_clip: float = 0.0) -> Tuple[float, Optional[float]]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    n = 0

    for batch in loader:
        x, y, _ = unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        if task_type == "classification":
            total_correct += (out.argmax(dim=1) == y).float().sum().item()

    avg_loss = float(total_loss / max(n, 1))
    if task_type == "classification":
        return avg_loss, float(total_correct / max(n, 1))
    return avg_loss, None


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, task_type: str, measure_throughput: bool = False):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    n = 0
    start = time.time()

    for batch in loader:
        x, y, _ = unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        if task_type == "classification":
            total_correct += (out.argmax(dim=1) == y).float().sum().item()

    end = time.time()
    avg_loss = float(total_loss / max(n, 1))
    acc = None
    if task_type == "classification":
        acc = float(total_correct / max(n, 1))
    throughput = None
    if measure_throughput and n > 0:
        throughput = float(n / max(end - start, 1e-6))
    return avg_loss, acc, throughput
