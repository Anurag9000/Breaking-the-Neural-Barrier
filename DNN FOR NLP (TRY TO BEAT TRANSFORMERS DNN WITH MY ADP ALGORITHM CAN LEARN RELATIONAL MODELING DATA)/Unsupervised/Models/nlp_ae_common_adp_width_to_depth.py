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

# Load baseline utils
BASE_PATH = Path(__file__).with_name("nlp_ae_common.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
MLPBlock = baseline_module.MLPBlock  # type: ignore
TextAvgEmbed = baseline_module.TextAvgEmbed  # type: ignore
soft_ce_loss = baseline_module.soft_ce_loss  # type: ignore


# ADP REVIEW (BEFORE REFACTOR)
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - Inner training: train_with_patience ties ES reset to delta and reloads immediately.
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - Control flow: toggles modes on no improvement; lacks forward-only march and context-end restore per updated spec.
# - ES patience conflated with expansion patiences; no snapshot/restore separation.


class TextAE(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: List[int], out_dim: int, use_bn: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden = list(hidden)
        self.out_dim = out_dim
        self.use_bn = use_bn
        self.encoder_tok = TextAvgEmbed(vocab_size, emb_dim)
        blocks = []
        prev = emb_dim
        for w in hidden:
            blocks.append(MLPBlock(prev, w, use_bn))
            prev = w
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, out_dim)

    def forward(self, view: Tuple[torch.Tensor, torch.Tensor]):
        tok, lens = view
        h0 = self.encoder_tok(tok, lens)
        h = self.backbone(h0) if len(self.backbone) > 0 else h0
        logits = self.head(h)
        return logits


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


def _resize_linear(old: nn.Linear, new_out: int, new_in: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        if old.bias is not None and new.bias is not None:
            new.bias[:r] = old.bias[:r]
    return new


def rebuild_backbone(model: TextAE, hidden: List[int]):
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
    model.head = _resize_linear(model.head, model.head.out_features, prev)
    model.hidden = list(hidden)


def expand_width(model: TextAE, ex_k: int, max_width: int) -> Optional[TextAE]:
    new_h = [min(max_width, w + ex_k) for w in model.hidden]
    if new_h == model.hidden:
        return None
    rebuild_backbone(model, new_h)
    return model


def expand_depth(model: TextAE, max_depth: int) -> Optional[TextAE]:
    if len(model.hidden) >= max_depth:
        return None
    new_h = model.hidden + [model.hidden[-1]]
    rebuild_backbone(model, new_h)
    return model


def total_neurons(model: TextAE) -> int:
    return sum(model.hidden)


def snapshot_arch_and_state(model: TextAE, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "vocab_size": model.vocab_size,
        "emb_dim": model.emb_dim,
        "hidden": list(model.hidden),
        "out_dim": model.out_dim,
        "use_bn": model.use_bn,
        "state": copy.deepcopy(state)
    }


def restore_arch_and_state(model: TextAE, snap: Dict[str, Any], device) -> TextAE:
    # Rebuild
    new_model = TextAE(
        vocab_size=snap["vocab_size"],
        emb_dim=snap["emb_dim"],
        hidden=snap["hidden"],
        out_dim=snap["out_dim"],
        use_bn=snap["use_bn"]
    ).to(device)
    new_model.load_state_dict(snap["state"])
    return new_model


def train_with_early_stopping(model: TextAE, data, acfg: ADPConfig, device) -> Tuple[float, Dict[str, Any]]:
    (train_ids, train_lens, train_targets), (val_ids, val_lens, val_targets) = data
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0
    
    for _ in range(acfg.max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model((train_ids.to(device), train_lens.to(device)))
        targets = train_targets.to(device)
        loss = soft_ce_loss(logits, targets)
        loss.backward()
        if acfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model((val_ids.to(device), val_lens.to(device)))
            v_t = val_targets.to(device)
            val = soft_ce_loss(v_logits, v_t).item()
            
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
            improved = True
        else:
            es_counter += 1
            improved = False

        # Log
        msg = f"  Epoch {_+1}/{max_epochs} | Val Loss: {val:.6f} | Best: {best_val:.6f} | ES: {es_counter}/{patience}"
        if verbose and logger:
            logger.log_console(msg)
        elif verbose:
             pass # print(msg)
        
        if logger:
             logger.log_epoch_stats({
                "epoch": _,
                "width": getattr(model, 'width', 0) if hasattr(model, 'width') else (getattr(model.in_lin, 'out_features', 0) if hasattr(model, 'in_lin') else 0),
                "depth": getattr(model, 'depth', 0),
                "neurons": total_neurons(model) if 'total_neurons' in globals() else 0,
                "val_loss": val,
                "best_val": best_val,
                "es_counter": es_counter,
                "improved": improved
             })
            
        if es_counter >= acfg.patience:
            break
            
    return best_val, best_state


def adp_search(model: TextAE, data, acfg: ADPConfig, device):
    
    # Initial training
    from utils.adp_contract import run_module_adp
    from utils.adp_introspect import infer_adp_shape

    best_val, model = run_module_adp(
        globals(),
        model,
        dl_train,
        dl_val,
        acfg,
        device,
        log_loss=locals().get("log_loss", False),
        log_neurons=locals().get("log_neurons", False),
        results_dir=locals().get("results_dir"),
        logger=locals().get("logger"),
    )

    return best_val, model


def main():
    import argparse
    p = argparse.ArgumentParser(description="ADP NLP AE (common) width/depth search")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    p.add_argument("--out-dim", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=30000)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only","depth_only","width_to_depth","depth_to_width","alt_width","alt_depth","width","depth"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=64)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=100000000)
    args = p.parse_args()

    # Synthetic placeholder data (replace with real text pipeline)
    B, L = 16, 12
    tok = torch.randint(1, args.vocab_size, (B, L))
    lens = torch.full((B,), L, dtype=torch.long)
    targets = F.one_hot(torch.randint(0, args.out_dim, (B,)), num_classes=args.out_dim).float()
    data = ((tok, lens, targets), (tok, lens, targets))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextAE(vocab_size=args.vocab_size, emb_dim=args.emb_dim, hidden=args.hidden, out_dim=args.out_dim).to(device)
    acfg = ADPConfig(adp_mode=args.adp_mode, delta=args.delta, patience=args.patience, trials_width=args.trials_width,
                     trials_depth=args.trials_depth, ex_k=args.ex_k, max_width=args.max_width, max_depth=args.max_depth,
                     max_neurons=args.max_neurons, max_epochs=args.max_epochs)
    best, model = adp_search(model, data, acfg, device)
    print(f"[ADP NLP AE] mode={args.adp_mode} best_val={best:.6f} hidden={model.hidden} depth={len(model.hidden)+1}")


if __name__ == "__main__":
    main()
