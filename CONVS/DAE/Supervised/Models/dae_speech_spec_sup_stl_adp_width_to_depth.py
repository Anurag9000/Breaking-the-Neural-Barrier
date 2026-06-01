import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.adp_logging import ContinuousLogger  # type: ignore
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore
from utils.audio_benchmarks import make_speechcommands_loaders

from CONVS.DAE.Supervised.Models.dae_speech_spec_sup_stl import (
    SupDAESpeechSpec,
    sup_dae_speech_total_neurons,
)


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 256
    max_depth: int = 8
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 300
    lambda_recon: float = 1.0
    vocab_size: int = 30
    seed: int = 1337


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt


def _merge_state(
    new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for k, v in new_state.items():
        if k in old_state:
            ov = old_state[k]
            merged[k] = ov if ov.shape == v.shape else _resize_tensor(v.shape, ov)
        else:
            merged[k] = v
    return merged


def rebuild_model(base: int, depth: int, cfg: ADPConfig, device: torch.device, state=None) -> SupDAESpeechSpec:
    model = SupDAESpeechSpec(vocab_size=cfg.vocab_size, base=base, depth=depth).to(device)
    if state is not None:
        merged = _merge_state(model.state_dict(), state)
        model.load_state_dict(merged, strict=False)
    return model


def expand_width(
    model: SupDAESpeechSpec,
    ex_k: int,
    max_width: int,
    cfg: ADPConfig,
    device: torch.device,
) -> Optional[SupDAESpeechSpec]:
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return None
    return rebuild_model(new_w, model.depth, cfg, device, model.state_dict())


def expand_depth(
    model: SupDAESpeechSpec,
    max_depth: int,
    cfg: ADPConfig,
    device: torch.device,
) -> Optional[SupDAESpeechSpec]:
    new_d = min(max_depth, model.depth + 1)
    if new_d == model.depth:
        return None
    return rebuild_model(model.width, new_d, cfg, device, model.state_dict())


def snapshot_arch_and_state(model: SupDAESpeechSpec, state=None) -> Dict[str, Any]:
    st = state if state is not None else model.state_dict()
    return {
        "width": model.width,
        "depth": model.depth,
        "vocab_size": model.vocab_size,
        "state": copy.deepcopy(st),
    }


def restore_arch_and_state(snap: Dict[str, Any], cfg: ADPConfig, device: torch.device) -> SupDAESpeechSpec:
    m = SupDAESpeechSpec(vocab_size=snap["vocab_size"], base=snap["width"], depth=snap["depth"]).to(device)
    m.load_state_dict(snap["state"], strict=False)
    return m


def make_loaders(
    cfg: ADPConfig,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, int]:
    train_loader, val_loader, _, num_labels = make_speechcommands_loaders(
        root=Path("./data/SpeechCommands"),
        batch_size=batch_size,
        download=True,
        num_workers=num_workers,
    )

    def collate_batch(batch):
        specs, labels = zip(*batch)
        max_t = max(spec.size(0) for spec in specs)
        feat_dim = specs[0].size(1)
        specs_padded = torch.zeros(len(specs), 1, max_t, feat_dim, dtype=specs[0].dtype)
        for i, spec in enumerate(specs):
            specs_padded[i, 0, : spec.size(0), :] = spec
        lab_padded = torch.tensor([[int(label) + 1] for label in labels], dtype=torch.long)
        lab_len = torch.ones(len(labels), dtype=torch.long)
        return specs_padded, lab_padded, lab_len

    dl_train = DataLoader(
        train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch
    )
    dl_val = DataLoader(
        val_loader.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch
    )
    return dl_train, dl_val, num_labels + 1


def train_with_early_stopping(
    model: SupDAESpeechSpec,
    dl_train: DataLoader,
    dl_val: DataLoader,
    cfg: ADPConfig,
    device: torch.device,
    history: List[float],
    logger: Optional[ContinuousLogger] = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for spec, labels, lab_len in dl_train:
            spec = spec.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lab_len = lab_len.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            spec_rec, logits = model(spec)
            loss_recon = mse(spec_rec, spec)

            B, T, V = logits.shape
            logp = logits.log_softmax(-1).transpose(0, 1)
            input_len = torch.full((B,), T, dtype=torch.long, device=device)
            flat_labels = labels.view(-1)
            loss_ctc = ctc(logp, flat_labels, input_len, lab_len)

            loss = loss_ctc + cfg.lambda_recon * loss_recon
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total += float(loss.item()) * B
            n += B
        train_loss = total / max(n, 1)

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for spec, labels, lab_len in dl_val:
                spec = spec.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                lab_len = lab_len.to(device, non_blocking=True)
                spec_rec, logits = model(spec)
                loss_recon = mse(spec_rec, spec) / spec.size(0)
                B, T, V = logits.shape
                logp = logits.log_softmax(-1).transpose(0, 1)
                input_len = torch.full((B,), T, dtype=torch.long, device=device)
                flat_labels = labels.view(-1)
                loss_ctc = ctc(logp, flat_labels, input_len, lab_len)
                loss = loss_ctc + cfg.lambda_recon * loss_recon
                total += float(loss.item()) * B
                n += B
        val_loss = total / max(n, 1)
        history.append(val_loss)

        improved = val_loss < best_val - cfg.delta
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1

        msg = (
            f"  Epoch {epoch:03d}/{cfg.max_epochs} | "
            f"Train={train_loss:.6f} | Val={val_loss:.6f} | "
            f"Best={best_val:.6f} | ES={es}/{cfg.patience}"
        )
        if logger:
            logger.log_console(msg)
        else:
            print(msg)

        if es >= cfg.patience:
            if logger:
                logger.log_console(f"  Early stopping at epoch {epoch}")
            else:
                print(f"  Early stopping at epoch {epoch}")
            break

    return best_val, best_state


def adp_search(
    cfg: ADPConfig,
    device: torch.device,
    logger: ContinuousLogger,
    batch_size: int,
    num_workers: int,
) -> Tuple[float, SupDAESpeechSpec]:
    dl_train, dl_val, vocab_size = make_loaders(cfg, batch_size, num_workers)
    history: List[float] = []

    cfg.vocab_size = vocab_size
    start_model = SupDAESpeechSpec(vocab_size=cfg.vocab_size, base=logger.width, depth=logger.depth).to(device)
    neurons = sup_dae_speech_total_neurons(logger.width, logger.depth, cfg.vocab_size)
    logger.log_architecture(logger.width, logger.depth, neurons)

    best_val, best_state = train_with_early_stopping(
        start_model, dl_train, dl_val, cfg, device, history, logger
    )
    global_best_snap = snapshot_arch_and_state(start_model, best_state)
    global_best_val = best_val

    def try_width_only(snap: Dict[str, Any], best_so_far: float) -> Tuple[Dict[str, Any], float]:
        fail = 0
        hist: List[Tuple[int, float]] = []
        curr = snap
        while fail < cfg.trials_width:
            m = restore_arch_and_state(curr, cfg, device)
            widened = expand_width(m, cfg.ex_k, cfg.max_width, cfg, device)
            if widened is None:
                break
            w = widened.width
            if sup_dae_speech_total_neurons(w, widened.depth, cfg.vocab_size) > cfg.max_neurons:
                break
            v, s = train_with_early_stopping(
                widened, dl_train, dl_val, cfg, device, history, logger
            )
            hist.append((w, v))
            if v < best_so_far - cfg.delta:
                best_so_far = v
                curr = snapshot_arch_and_state(widened, s)
                fail = 0
            else:
                fail += 1
        if hist:
            logger.log_width_search(hist)
        return curr, best_so_far

    def try_depth_only(snap: Dict[str, Any], best_so_far: float) -> Tuple[Dict[str, Any], float]:
        fail = 0
        hist: List[Tuple[int, float]] = []
        curr = snap
        while fail < cfg.trials_depth:
            m = restore_arch_and_state(curr, cfg, device)
            deeper = expand_depth(m, cfg.max_depth, cfg, device)
            if deeper is None:
                break
            d = deeper.depth
            if sup_dae_speech_total_neurons(deeper.width, d, cfg.vocab_size) > cfg.max_neurons:
                break
            v, s = train_with_early_stopping(
                deeper, dl_train, dl_val, cfg, device, history, logger
            )
            hist.append((d, v))
            if v < best_so_far - cfg.delta:
                best_so_far = v
                curr = snapshot_arch_and_state(deeper, s)
                fail = 0
            else:
                fail += 1
        if hist:
            logger.log_depth_search(hist)
        return curr, best_so_far

    mode = cfg.adp_mode.lower()
    if mode == "width_only":
        global_best_snap, global_best_val = try_width_only(global_best_snap, global_best_val)
    elif mode == "depth_only":
        global_best_snap, global_best_val = try_depth_only(global_best_snap, global_best_val)
    elif mode == "width_to_depth":
        global_best_snap, global_best_val = try_width_only(global_best_snap, global_best_val)
        global_best_snap, global_best_val = try_depth_only(global_best_snap, global_best_val)
    elif mode == "depth_to_width":
        global_best_snap, global_best_val = try_depth_only(global_best_snap, global_best_val)
        global_best_snap, global_best_val = try_width_only(global_best_snap, global_best_val)
    elif mode == "alt_width":
        for _ in range(max(cfg.trials_width, cfg.trials_depth)):
            global_best_snap, global_best_val = try_width_only(global_best_snap, global_best_val)
            global_best_snap, global_best_val = try_depth_only(global_best_snap, global_best_val)
    elif mode == "alt_depth":
        for _ in range(max(cfg.trials_width, cfg.trials_depth)):
            global_best_snap, global_best_val = try_depth_only(global_best_snap, global_best_val)
            global_best_snap, global_best_val = try_width_only(global_best_snap, global_best_val)
    else:
        raise ValueError(f"Unknown adp_mode={cfg.adp_mode}")

    final_model = restore_arch_and_state(global_best_snap, cfg, device)
    return global_best_val, final_model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--adp-mode", type=str, default="width_to_depth")
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--trials-width", type=int, default=2)
    parser.add_argument("--trials-depth", type=int, default=2)
    parser.add_argument("--ex-k", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-width", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-neurons", type=int, default=5_000_000)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results_dae_speech_spec_sup_adp")
    parser.add_argument("--plot-loss", action="store_true")
    parser.add_argument("--plot-neurons", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = ADPConfig(
        adp_mode=args.adp_mode,
        delta=args.delta,
        patience=args.patience,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
        ex_k=args.ex_k,
        max_width=args.max_width,
        max_depth=args.max_depth,
        max_neurons=args.max_neurons,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_epochs=args.max_epochs,
        lambda_recon=args.lambda_recon,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )

    logger = ContinuousLogger(
        experiment_name="dae_speech_spec_sup",
        mode=args.adp_mode,
        results_dir=results_dir,
    )
    logger.batch_size = args.batch_size
    logger.num_workers = args.num_workers
    logger.width = args.width
    logger.depth = args.depth

    best_val, _ = adp_search(
        cfg=cfg,
        device=device,
        logger=logger,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    logger.log_final(best_val)

    if args.plot_loss or args.plot_neurons:
        stats_csv = results_dir / "training_stats.csv"
        if stats_csv.exists():
            plot_loss_vs_epoch(stats_csv, results_dir / "loss_vs_epoch.png")
            plot_loss_vs_neurons(stats_csv, results_dir / "loss_vs_neurons.png")
