import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load baseline
BASE_PATH = Path(__file__).with_name("nlp_ssl_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
MLPTextSSL = baseline_module.MLPTextSSL  # type: ignore
MLPBlock = baseline_module.MLPBlock  # type: ignore
TextAvgEmbed = baseline_module.TextAvgEmbed  # type: ignore
nt_xent_loss = baseline_module.nt_xent_loss  # type: ignore


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 10
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 64
    max_width: int = 4096
    max_depth: int = 12
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 30
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


def rebuild_backbone(model: MLPTextSSL, hidden):
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
    # proj first layer must match rep_dim
    model.proj[0] = _resize_linear(model.proj[0], model.proj[0].out_features, model.rep.out_features)
    model.hidden = list(hidden)


def widen_all(model: MLPTextSSL, ex_k: int, max_width: int):
    new_h = [min(max_width, w + ex_k) for w in model.hidden]
    rebuild_backbone(model, new_h)


def append_depth(model: MLPTextSSL):
    new_h = model.hidden + [model.hidden[-1]]
    rebuild_backbone(model, new_h)


def total_neurons(model: MLPTextSSL) -> int:
    return sum(model.hidden)


def train_with_patience(model: MLPTextSSL, data, acfg: ADPConfig, device):
    (train_v1, train_v2), (val_v1, val_v2) = data
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best = float("inf")
    best_state = None
    pat = acfg.patience
    for _ in range(acfg.max_epochs):
        model.train()
        total, n = 0.0, 0
        for (ids1, len1), (ids2, len2) in zip(train_v1, train_v2):
            ids1, len1 = ids1.to(device), len1.to(device)
            ids2, len2 = ids2.to(device), len2.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model((ids1, len1), (ids2, len2), temperature=acfg.temperature)
            loss.backward()
            if acfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
            total += loss.item()
            n += 1
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            m = 0
            for (ids1, len1), (ids2, len2) in zip(val_v1, val_v2):
                ids1, len1 = ids1.to(device), len1.to(device)
                ids2, len2 = ids2.to(device), len2.to(device)
                val_loss += model((ids1, len1), (ids2, len2), temperature=acfg.temperature).item()
                m += 1
            val_loss /= max(m, 1)
        if val_loss < best - acfg.delta:
            best = val_loss
            best_state = copy.deepcopy(model.state_dict())
            pat = acfg.patience
        else:
            pat -= 1
        if pat <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def adp_search(model: MLPTextSSL, data, acfg: ADPConfig, device):
    def can_widen():
        return max(model.hidden) + acfg.ex_k <= acfg.max_width and total_neurons(model) < acfg.max_neurons

    def can_deepen():
        return len(model.hidden) + 1 <= acfg.max_depth and (total_neurons(model) + model.hidden[-1]) <= acfg.max_neurons

    inner_val = train_with_patience(model, data, acfg, device)
    best_val = inner_val
    best_state = copy.deepcopy(model.state_dict())

    pw = acfg.trials_width
    pd = acfg.trials_depth
    mode = acfg.adp_mode
    improved = True
    while improved:
        improved = False
        if mode in ("width_only", "width", "width_to_depth", "alt_width"):
            if can_widen() and pw > 0:
                pre = copy.deepcopy(model.state_dict())
                pre_val = inner_val
                widen_all(model, acfg.ex_k, acfg.max_width)
                v = train_with_patience(model, data, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pw = acfg.trials_width
                    improved = True
                    if v < best_val:
                        best_val = v
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(pre)
                    pw -= 1
            if mode == "width_only":
                continue
        if mode in ("depth_only", "depth", "depth_to_width", "alt_depth"):
            if can_deepen() and pd > 0:
                pre = copy.deepcopy(model.state_dict())
                pre_val = inner_val
                append_depth(model)
                v = train_with_patience(model, data, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pd = acfg.trials_depth
                    improved = True
                    if v < best_val:
                        best_val = v
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(pre)
                    pd -= 1
            if mode == "depth_only":
                continue
        if mode == "width_to_depth" and not improved:
            mode = "depth"
            pd = acfg.trials_depth
            improved = True
        elif mode == "depth_to_width" and not improved:
            mode = "width"
            pw = acfg.trials_width
            improved = True
        elif mode in ("alt_width", "alt_depth"):
            mode = "depth" if mode == "alt_width" else "width"
            improved = True

    model.load_state_dict(best_state)
    return best_val


def main():
    import argparse
    p = argparse.ArgumentParser(description="ADP NLP SSL (MLP) width/depth search")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    p.add_argument("--rep-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=30000)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=64)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    # Placeholder synthetic data (replace with real text pipeline + collate)
    B, L = 16, 12
    tok = torch.randint(1, args.vocab_size, (B, L))
    lens = torch.full((B,), L, dtype=torch.long)
    v1 = [((tok, lens))] * 4
    v2 = [((tok, lens))] * 4
    data = ((v1, v2), (v1, v2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPTextSSL(vocab_size=args.vocab_size, emb_dim=args.emb_dim, hidden=args.hidden,
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
    best = adp_search(model, data, acfg, device)
    print(f"[ADP NLP SSL] mode={args.adp_mode} best_val={best:.6f} hidden={model.hidden} depth={len(model.hidden)+1}")


if __name__ == "__main__":
    main()
