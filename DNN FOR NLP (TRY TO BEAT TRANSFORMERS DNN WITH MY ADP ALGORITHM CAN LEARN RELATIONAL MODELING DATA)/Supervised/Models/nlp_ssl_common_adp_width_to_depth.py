import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore
from utils.adp_logging import ContinuousLogger
from utils.text_benchmarks import make_ag_news_ssl_loaders

# Load baseline utils
BASE_PATH = Path(__file__).with_name("nlp_ssl_common.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
MLPBlock = baseline_module.MLPBlock  # type: ignore
TextAvgEmbed = baseline_module.TextAvgEmbed  # type: ignore
nt_xent_loss = baseline_module.nt_xent_loss  # type: ignore


# ADP REVIEW (BEFORE REFACTOR)
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - Inner training: train_with_patience ties ES reset to delta and reloads immediately.
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - Control flow: toggles modes on no improvement; lacks forward-only march and context-end restore per updated spec.
# - ES patience conflated with expansion patiences; no snapshot/restore separation.


class MLPTextSSL(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: List[int], rep_dim: int, proj_dim: int, use_bn: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden = list(hidden)
        self.rep_dim = rep_dim
        self.proj_dim = proj_dim
        self.use_bn = use_bn
        self.encoder_tok = TextAvgEmbed(vocab_size, emb_dim)
        blocks = []
        prev = emb_dim
        for w in hidden:
            blocks.append(MLPBlock(prev, w, use_bn))
            prev = w
        self.backbone = nn.Sequential(*blocks)
        self.rep = nn.Linear(prev, rep_dim)
        self.proj = nn.Sequential(
            nn.Linear(rep_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def encode(self, view: Tuple[torch.Tensor, torch.Tensor]):
        tok, lens = view
        h0 = self.encoder_tok(tok, lens)
        h = self.backbone(h0) if len(self.backbone) > 0 else h0
        z = self.rep(h)
        p = self.proj(z)
        return p

    def forward(self, v1, v2, temperature=0.05):
        z1 = self.encode(v1)
        z2 = self.encode(v2)
        return nt_xent_loss(z1, z2, temperature=temperature)


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 64
    max_width: int = 4096
    max_depth: int = 12
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 100_000_000
    temperature: float = 0.05


def _resize_linear(old: nn.Linear, new_out: int, new_in: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        if old.bias is not None and new.bias is not None:
            new.bias[:r] = old.bias[:r]
    return new


def rebuild_backbone(model: MLPTextSSL, hidden: List[int]):
    device = next(model.parameters()).device
    use_bn = model.use_bn
    blocks = []
    prev = model.emb_dim
    old_blocks = list(model.backbone)
    for w in hidden:
        blk = MLPBlock(prev, w, use_bn).to(device)
        if old_blocks:
            old_blk = old_blocks.pop(0)
            blk.linear = _resize_linear(old_blk.linear, w, prev)
        blocks.append(blk)
        prev = w
    model.backbone = nn.Sequential(*blocks)
    model.rep = _resize_linear(model.rep, model.rep.out_features, prev)
    model.proj[0] = _resize_linear(model.proj[0], model.proj[0].out_features, model.rep.out_features)
    model.hidden = list(hidden)


def expand_width(model: MLPTextSSL, ex_k: int, max_width: int) -> Optional[MLPTextSSL]:
    new_h = [min(max_width, w + ex_k) for w in model.hidden]
    if new_h == model.hidden:
        return None
    rebuild_backbone(model, new_h)
    return model


def expand_depth(model: MLPTextSSL, max_depth: int) -> Optional[MLPTextSSL]:
    if len(model.hidden) >= max_depth:
        return None
    new_h = model.hidden + [model.hidden[-1]]
    rebuild_backbone(model, new_h)
    return model


def total_neurons(model: MLPTextSSL) -> int:
    return sum(model.hidden)


def snapshot_arch_and_state(model: MLPTextSSL, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "vocab_size": model.vocab_size,
        "emb_dim": model.emb_dim,
        "hidden": list(model.hidden),
        "rep_dim": model.rep_dim,
        "proj_dim": model.proj_dim,
        "use_bn": model.use_bn,
        "state": copy.deepcopy(state)
    }


def restore_arch_and_state(model: MLPTextSSL, snap: Dict[str, Any], device) -> MLPTextSSL:
    # Rebuild
    new_model = MLPTextSSL(
        vocab_size=snap["vocab_size"],
        emb_dim=snap["emb_dim"],
        hidden=snap["hidden"],
        rep_dim=snap["rep_dim"],
        proj_dim=snap["proj_dim"],
        use_bn=snap["use_bn"]
    ).to(device)
    new_model.load_state_dict(snap["state"])
    return new_model


def train_with_early_stopping(
    model: MLPTextSSL,
    dl_train,
    dl_val,
    acfg: ADPConfig,
    device,
    history,
) -> Tuple[float, Dict[str, Any]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for _ in range(acfg.max_epochs):
        model.train()
        for (ids1, len1), (ids2, len2) in dl_train:
            ids1, len1 = ids1.to(device), len1.to(device)
            ids2, len2 = ids2.to(device), len2.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model((ids1, len1), (ids2, len2), temperature=acfg.temperature)
            loss.backward()
            if acfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
            
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            m = 0
            for (ids1, len1), (ids2, len2) in dl_val:
                ids1, len1 = ids1.to(device), len1.to(device)
                ids2, len2 = ids2.to(device), len2.to(device)
                val_loss += model((ids1, len1), (ids2, len2), temperature=acfg.temperature).item()
                m += 1
            val_loss /= max(m, 1)
            
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1

        history.append(val_loss)
            
        if es_counter >= acfg.patience:
            break
            
    return best_val, best_state


def adp_search(model: MLPTextSSL, data, acfg: ADPConfig, device, results_dir: Path | None = None, logger: ContinuousLogger | None = None):
    dl_train, dl_val = data

    from utils.adp_contract import run_module_adp

    best_val, model = run_module_adp(
        globals(),
        model,
        dl_train,
        dl_val,
        acfg,
        device,
        log_loss=False,
        log_neurons=False,
        results_dir=results_dir,
        logger=logger,
    )
    return best_val, model


def main():
    import argparse
    p = argparse.ArgumentParser(description="ADP NLP SSL (common) width/depth search")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    p.add_argument("--rep-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=30000)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=64)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    trl, val, test, vocab = make_ag_news_ssl_loaders(
        batch_size=args.batch_size,
        max_len=128,
        seed=0,
        val_fraction=0.1,
        word_dropout=0.1,
        min_freq=2,
        max_vocab=args.vocab_size,
    )
    data = (trl, val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPTextSSL(vocab_size=len(vocab), emb_dim=args.emb_dim, hidden=args.hidden,
                       rep_dim=args.rep_dim, proj_dim=args.proj_dim).to(device)
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
        max_epochs=args.max_epochs,
    )
    best, model = adp_search(model, data, acfg, device)
    print(f"[ADP NLP SSL COMMON] mode={args.adp_mode} best_val={best:.6f} hidden={model.hidden} depth={len(model.hidden)+1}")
