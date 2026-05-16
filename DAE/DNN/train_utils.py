import json
import subprocess
import threading
import time
from pathlib import Path
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


def _batchnorm_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def forward_train_safe(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    batchnorm_layers = _batchnorm_modules(model)
    if x.size(0) != 1 or not batchnorm_layers:
        return model(x)

    prior_states = [(layer, layer.training) for layer in batchnorm_layers]
    for layer, _ in prior_states:
        layer.eval()
    try:
        return model(x)
    finally:
        for layer, was_training in prior_states:
            layer.train(was_training)


def query_gpu_vram_used_mb(device_index: int = 0) -> Optional[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={int(device_index)}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    text = out.decode("utf-8").strip().splitlines()
    if not text:
        return None
    try:
        return int(float(text[0].strip()))
    except Exception:
        return None


class AdaptiveBatchController:
    def __init__(
        self,
        initial_batch_size: int,
        *,
        threshold_gb: float = 5.5,
        poll_interval_sec: float = 30.0,
        shrink_factor: float = 0.75,
        min_batch_size: int = 1,
        state_path: Optional[Path] = None,
    ):
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.threshold_mb = float(threshold_gb) * 1024.0
        self.poll_interval_sec = float(poll_interval_sec)
        self.shrink_factor = float(shrink_factor)
        self.min_batch_size = int(min_batch_size)
        self.state_path = Path(state_path) if state_path is not None else None
        self._current_batch_size = max(self.min_batch_size, int(initial_batch_size))
        self._last_poll = 0.0
        self._last_vram_mb: Optional[int] = None
        self._device_index = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        self._persist_state()

    @property
    def current_batch_size(self) -> int:
        with self._lock:
            return int(self._current_batch_size)

    @property
    def last_vram_mb(self) -> Optional[int]:
        with self._lock:
            return self._last_vram_mb

    def _persist_state(self) -> None:
        if self.state_path is None:
            return
        payload = {
            "batch_size": int(self.current_batch_size),
            "threshold_gb": float(self.threshold_mb / 1024.0),
            "poll_interval_sec": float(self.poll_interval_sec),
            "shrink_factor": float(self.shrink_factor),
            "last_vram_mb": self.last_vram_mb,
            "timestamp": time.time(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _shrink_locked(self, reason: str) -> int:
        new_batch = max(self.min_batch_size, int(self._current_batch_size * self.shrink_factor))
        if new_batch < self._current_batch_size:
            self._current_batch_size = new_batch
        self._persist_state()
        return self._current_batch_size

    def maybe_poll(self, force: bool = False) -> Optional[int]:
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_poll) < self.poll_interval_sec:
                return None
            self._last_poll = now
            used_mb = query_gpu_vram_used_mb(self._device_index)
            self._last_vram_mb = used_mb
            if used_mb is None:
                self._persist_state()
                return None
            if float(used_mb) > float(self.threshold_mb):
                return self._shrink_locked("vram")
            self._persist_state()
            return None

    def start(self) -> None:
        if self._thread is not None:
            return

        def _loop():
            while not self._stop_event.wait(self.poll_interval_sec):
                self.maybe_poll(force=True)

        self._thread = threading.Thread(target=_loop, name="adaptive-batch-controller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def reduce_after_oom(self) -> int:
        with self._lock:
            return self._shrink_locked("oom")


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
        out = forward_train_safe(model, x)
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
        out = forward_train_safe(model, x)
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
