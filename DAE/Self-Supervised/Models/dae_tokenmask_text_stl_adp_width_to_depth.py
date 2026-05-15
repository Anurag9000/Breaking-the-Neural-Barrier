import copy
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger  # type: ignore
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

from .dae_tokenmask_text_stl import TokenMaskTransformerDAE, token_dae_total_neurons


SPECIAL_TOKENS = ["<pad>", "<unk>", "<mask>"]


class TextDataset(Dataset):
    def __init__(self, lines: List[str], vocab: dict, max_len: int):
        self.lines = lines
        self.vocab = vocab
        self.max_len = max_len
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]

    def __len__(self) -> int:
        return len(self.lines)

    def _encode(self, text: str) -> torch.Tensor:
        tokens = text.strip().split()
        ids = [self.vocab.get(tok, self.unk_id) for tok in tokens][: self.max_len]
        if len(ids) < self.max_len:
            ids.extend([self.pad_id] * (self.max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._encode(self.lines[idx])


def build_vocab(lines: List[str], vocab_size: int) -> dict:
    from collections import Counter

    counter: Counter = Counter()
    for line in lines:
        counter.update(line.strip().split())
    most_common = [w for w, _ in counter.most_common(max(vocab_size - len(SPECIAL_TOKENS), 0))]
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for w in most_common:
        vocab[w] = len(vocab)
    return vocab


def random_mask_tokens(
    input_ids: torch.Tensor,
    mask_id: int,
    pad_id: int,
    mask_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask_prob <= 0.0:
        return input_ids, torch.full_like(input_ids, pad_id)
    B, L = input_ids.shape
    rand = torch.rand((B, L), device=input_ids.device)
    mask = (rand < mask_prob) & input_ids.ne(pad_id)

    masked = input_ids.clone()
    masked[mask] = mask_id

    target = torch.full_like(input_ids, pad_id)
    target[mask] = input_ids[mask]
    return masked, target


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 64
    max_width: int = 512  # interpreted as max d_model
    max_depth: int = 16
    max_neurons: int = 10_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 100_000_000
    mask_prob: float = 0.15
    max_len: int = 64
    vocab_size: int = 20000


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt


def _merge_state(
    new_state: Dict[str, torch.Tensor],
    old_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for k, v in new_state.items():
        if k in old_state:
            ov = old_state[k]
            merged[k] = ov if ov.shape == v.shape else _resize_tensor(v.shape, ov)
        else:
            merged[k] = v
    return merged


def rebuild_model(
    model: TokenMaskTransformerDAE,
    d_model: int,
    depth: int,
    device: torch.device,
) -> TokenMaskTransformerDAE:
    new_model = TokenMaskTransformerDAE(
        vocab_size=model.vocab_size,
        d_model=d_model,
        depth=depth,
        num_heads=model.num_heads,
        dim_feedforward=model.dim_feedforward,
        max_len=model.max_len,
        pad_id=model.pad_id,
    ).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(
    model: TokenMaskTransformerDAE,
    ex_k: int,
    max_width: int,
    device: torch.device,
) -> Optional[TokenMaskTransformerDAE]:
    new_dim = min(max_width, model.d_model + ex_k)
    if new_dim == model.d_model:
        return None
    return rebuild_model(model, new_dim, model.depth, device)


def expand_depth(
    model: TokenMaskTransformerDAE,
    max_depth: int,
    device: torch.device,
) -> Optional[TokenMaskTransformerDAE]:
    if model.depth >= max_depth:
        return None
    return rebuild_model(model, model.d_model, model.depth + 1, device)


def snapshot_arch_and_state(
    model: TokenMaskTransformerDAE,
    state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    st = state if state is not None else model.state_dict()
    return {
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "depth": model.depth,
        "num_heads": model.num_heads,
        "dim_feedforward": model.dim_feedforward,
        "max_len": model.max_len,
        "pad_id": model.pad_id,
        "state": copy.deepcopy(st),
    }


def restore_arch_and_state(snap: Dict[str, Any], device: torch.device) -> TokenMaskTransformerDAE:
    mdl = TokenMaskTransformerDAE(
        vocab_size=snap["vocab_size"],
        d_model=snap["d_model"],
        depth=snap["depth"],
        num_heads=snap["num_heads"],
        dim_feedforward=snap["dim_feedforward"],
        max_len=snap["max_len"],
        pad_id=snap["pad_id"],
    ).to(device)
    mdl.load_state_dict(snap["state"], strict=False)
    return mdl


def train_with_early_stopping(
    model: TokenMaskTransformerDAE,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    mask_id: int,
    pad_id: int,
    history: List[float],
    logger: Optional[ContinuousLogger] = None,
    verbose: bool = True,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for epoch in range(1, acfg.max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for ids in dl_train:
            ids = ids.to(device, non_blocking=True)
            masked_ids, targets = random_mask_tokens(ids, mask_id, pad_id, acfg.mask_prob)
            logits, _ = model(masked_ids)
            loss = ce(logits.view(-1, logits.size(-1)), targets.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if acfg.grad_clip is not None and acfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()

            total += float(loss.item()) * ids.size(0)
            n += ids.size(0)
        train_loss = total / max(n, 1)

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for ids in dl_val:
                ids = ids.to(device, non_blocking=True)
                masked_ids, targets = random_mask_tokens(ids, mask_id, pad_id, acfg.mask_prob)
                logits, _ = model(masked_ids)
                loss = ce(logits.view(-1, logits.size(-1)), targets.view(-1))
                total += float(loss.item())
                n += ids.size(0)
        val_loss = total / max(n, 1)
        history.append(val_loss)

        improved = val_loss < best_val - acfg.delta
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1

        msg = (
            f"  Epoch {epoch:03d}/{acfg.max_epochs} | "
            f"Train={train_loss:.6f} | Val={val_loss:.6f} | "
            f"Best={best_val:.6f} | ES={es_counter}/{acfg.patience}"
        )
        if logger:
            logger.log_console(msg)
        elif verbose:
            print(msg)

        if es_counter >= acfg.patience:
            if logger:
                logger.log_console(f"  Early stopping at epoch {epoch}")
            elif verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    return best_val, best_state


def make_loaders(
    text_path: Path,
    acfg: ADPConfig,
    batch_size: int,
    val_frac: float,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, dict]:
    lines = [ln.strip() for ln in text_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"No non-empty lines found in {text_path}")
    vocab = build_vocab(lines, acfg.vocab_size)
    dataset = TextDataset(lines, vocab, acfg.max_len)
    n_total = len(dataset)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, vocab


def adp_search(
    model: TokenMaskTransformerDAE,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    mask_id: int,
    pad_id: int,
    logger: ContinuousLogger,
    results_dir: Path,
    log_loss: bool,
    log_neurons: bool,
) -> Tuple[float, TokenMaskTransformerDAE, int, int]:
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    improvements: List[Tuple[int, float]] = []

    base_val, base_state = train_with_early_stopping(
        model, dl_train, dl_val, acfg, device, mask_id, pad_id, val_history, logger=logger, verbose=True
    )
    best_snap = snapshot_arch_and_state(model, base_state)
    global_best_val = base_val
    global_best_snap = copy.deepcopy(best_snap)
    best_d_model, best_depth = model.d_model, model.depth
    improvements.append((token_dae_total_neurons(best_d_model, best_depth), global_best_val))

    def optimize_width_at_fixed_depth(
        start_snap: Dict[str, Any],
        current_best_val: float,
    ) -> Tuple[Dict[str, Any], float]:
        nonlocal improvements
        snap = copy.deepcopy(start_snap)
        fail = 0
        while fail < acfg.trials_width:
            curr_model = restore_arch_and_state(snap, device)
            wider = expand_width(curr_model, acfg.ex_k, acfg.max_width, device)
            if wider is None or token_dae_total_neurons(wider.d_model, wider.depth) > acfg.max_neurons:
                break
            logger.log_console(
                f"[WIDTH OPT] Trying d_model={wider.d_model}, depth={wider.depth}, "
                f"neurons={token_dae_total_neurons(wider.d_model, wider.depth)}"
            )
            val, state = train_with_early_stopping(
                wider, dl_train, dl_val, acfg, device, mask_id, pad_id, val_history, logger=logger, verbose=False
            )
            if val + acfg.delta < current_best_val:
                current_best_val = val
                snap = snapshot_arch_and_state(wider, state)
                improvements.append((token_dae_total_neurons(wider.d_model, wider.depth), val))
                logger.log_console(
                    f"[WIDTH OPT] ✓ IMPROVEMENT: d_model={wider.d_model}, depth={wider.depth}, val={val:.6f}"
                )
                fail = 0
            else:
                fail += 1
                logger.log_console(
                    f"[WIDTH OPT] ✗ No improvement: d_model={wider.d_model}, depth={wider.depth}, val={val:.6f}"
                )
        return snap, current_best_val

    def optimize_depth_at_fixed_width(
        start_snap: Dict[str, Any],
        current_best_val: float,
    ) -> Tuple[Dict[str, Any], float]:
        nonlocal improvements
        snap = copy.deepcopy(start_snap)
        fail = 0
        while fail < acfg.trials_depth:
            curr_model = restore_arch_and_state(snap, device)
            deeper = expand_depth(curr_model, acfg.max_depth, device)
            if deeper is None or token_dae_total_neurons(deeper.d_model, deeper.depth) > acfg.max_neurons:
                break
            logger.log_console(
                f"[DEPTH OPT] Trying d_model={deeper.d_model}, depth={deeper.depth}, "
                f"neurons={token_dae_total_neurons(deeper.d_model, deeper.depth)}"
            )
            val, state = train_with_early_stopping(
                deeper, dl_train, dl_val, acfg, device, mask_id, pad_id, val_history, logger=logger, verbose=False
            )
            if val + acfg.delta < current_best_val:
                current_best_val = val
                snap = snapshot_arch_and_state(deeper, state)
                improvements.append((token_dae_total_neurons(deeper.d_model, deeper.depth), val))
                logger.log_console(
                    f"[DEPTH OPT] ✓ IMPROVEMENT: d_model={deeper.d_model}, depth={deeper.depth}, val={val:.6f}"
                )
                fail = 0
            else:
                fail += 1
                logger.log_console(
                    f"[DEPTH OPT] ✗ No improvement: d_model={deeper.d_model}, depth={deeper.depth}, val={val:.6f}"
                )
        return snap, current_best_val

    mode = acfg.adp_mode
    logger.log_console(f"[ADP] Starting search mode={mode}")

    if mode == "width_only":
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
    elif mode == "depth_only":
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
    elif mode == "width_to_depth":
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
    elif mode == "depth_to_width":
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
    elif mode == "alt_width":
        turn_width = True
        while True:
            prev_val = global_best_val
            if turn_width:
                global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
            else:
                global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
            if abs(global_best_val - prev_val) < acfg.delta:
                break
            turn_width = not turn_width
    elif mode == "alt_depth":
        turn_width = False
        while True:
            prev_val = global_best_val
            if turn_width:
                global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
            else:
                global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
            if abs(global_best_val - prev_val) < acfg.delta:
                break
            turn_width = not turn_width
    else:
        raise ValueError(f"Unknown adp_mode: {mode}")

    best_model = restore_arch_and_state(global_best_snap, device)
    best_d_model, best_depth = best_model.d_model, best_model.depth

    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png")
    if log_neurons and improvements:
        ns = [n for n, _ in improvements]
        vs = [v for _, v in improvements]
        plot_loss_vs_neurons(ns, vs, results_dir / "loss_vs_neurons.png")

    return global_best_val, best_model, best_d_model, best_depth


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--text-path", type=str, required=True)
    p.add_argument("--results-dir", type=str, default="results_adp_dae_tokenmask_text")
    p.add_argument("--adp-mode", type=str, default="width_to_depth")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--ex-k", type=int, default=64)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=16)
    p.add_argument("--max-neurons", type=int, default=10000000)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=20000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = ContinuousLogger(
        results_dir / "training_log.txt",
        console_prefix="[ADP DAE TokenMask]",
    )
    logger.log_console(f"Initialized ADP search (mode={args.adp_mode})")

    acfg = ADPConfig(
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
        max_epochs=args.max_epochs,
        mask_prob=args.mask_prob,
        max_len=args.max_len,
        vocab_size=args.vocab_size,
    )

    text_path = Path(args.text_path)
    dl_train, dl_val, vocab = make_loaders(
        text_path=text_path,
        acfg=acfg,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    pad_id = vocab["<pad>"]
    mask_id = vocab["<mask>"]

    base_model = TokenMaskTransformerDAE(
        vocab_size=len(vocab),
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        dim_feedforward=args.ffn_dim,
        max_len=args.max_len,
        pad_id=pad_id,
    ).to(device)
    logger.log_console(
        f"Base model: d_model={base_model.d_model}, depth={base_model.depth}, "
        f"neurons={token_dae_total_neurons(base_model.d_model, base_model.depth)}"
    )

    best_val, best_model, best_d_model, best_depth = adp_search(
        base_model,
        dl_train,
        dl_val,
        acfg,
        device,
        mask_id,
        pad_id,
        logger,
        results_dir,
        log_loss=args.plot_loss,
        log_neurons=args.plot_neurons,
    )

    logger.log_console(
        f"[ADP DONE] Best val={best_val:.6f} at d_model={best_d_model}, depth={best_depth}, "
        f"neurons={token_dae_total_neurons(best_d_model, best_depth)}"
    )

    torch.save(
        {
            "model": best_model.state_dict(),
            "d_model": best_d_model,
            "depth": best_depth,
            "vocab": vocab,
            "best_val": best_val,
            "config": vars(args),
        },
        results_dir / "best_adp_model.pt",
    )
