import copy
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Load baseline
BASELINE_PATH = Path(__file__).with_name("dnn_ae_graph.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASELINE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
DNNNodeAE = baseline_module.DNNNodeAE  # type: ignore
TrainCfg = baseline_module.TrainCfg  # type: ignore
load_planetoid = baseline_module.load_planetoid  # type: ignore


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"  # {"width_only","depth_only","width_to_depth","depth_to_width","alt_width","alt_depth","width","depth"}
    delta: float = 1e-3
    patience: int = 100
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 32
    max_width: int = 4096
    max_depth: int = 16
    max_neurons: int = 5_000_000


def _resize_linear(old: nn.Linear, new_out: int, new_in: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        if old.bias is not None and new.bias is not None:
            new.bias[:r] = old.bias[:r]
    return new


def total_neurons(model: DNNNodeAE) -> int:
    h = model.in_lin.out_features
    return h * (len(model.hiddens) + 1)


def widen_all(model: DNNNodeAE, ex_k: int, max_width: int):
    """Increase hidden width everywhere by ex_k (capped)."""
    new_h = min(max_width, model.in_lin.out_features + ex_k)
    model.in_lin = _resize_linear(model.in_lin, new_h, model.in_lin.in_features)
    prev = new_h
    new_hiddens = nn.ModuleList()
    for lin in model.hiddens:
        nh = min(max_width, lin.out_features + ex_k)
        new_hiddens.append(_resize_linear(lin, nh, prev))
        prev = nh
    model.hiddens = new_hiddens
    # decoder_out may be None until forward; if exists, resize input
    if model.decoder_out is not None:
        model.decoder_out = _resize_linear(model.decoder_out, model.decoder_out.out_features, prev)
    model.hidden = prev


def append_depth(model: DNNNodeAE):
    """Append one hidden layer (square) before decoder_out."""
    width = model.hidden
    device = model.in_lin.weight.device
    model.hiddens.append(nn.Linear(width, width, bias=False).to(device))
    if model.decoder_out is not None:
        model.decoder_out = _resize_linear(model.decoder_out, model.decoder_out.out_features, width)
    model.depth = len(model.hiddens) + 1


def train_ae(model: DNNNodeAE, data, cfg: TrainCfg, patience: int, max_epochs: int) -> float:
    X, _, train_mask, val_mask, _ = data
    X = X.to(cfg.device)
    train_mask = train_mask.to(cfg.device)
    val_mask = val_mask.to(cfg.device)
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf")
    best_state = None
    pat = patience
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        Xh = model(X)
        loss = F.mse_loss(Xh[train_mask], X[train_mask])
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        # val
        model.eval()
        with torch.no_grad():
            val = F.mse_loss(model(X)[val_mask], X[val_mask]).item()
        if val < best - 1e-9:
            best = val
            best_state = copy.deepcopy(model.state_dict())
            pat = patience
        else:
            pat -= 1
        if pat <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def adp_search(model: DNNNodeAE, data, tcfg: TrainCfg, acfg: ADPConfig):
    """Unified ADP search across width/depth policies."""
    def can_widen():
        return model.in_lin.out_features + acfg.ex_k <= acfg.max_width and total_neurons(model) < acfg.max_neurons

    def can_deepen():
        return (len(model.hiddens) + 1) < acfg.max_depth and (total_neurons(model) + model.hidden) <= acfg.max_neurons

    inner_val = train_ae(model, data, tcfg, patience=acfg.patience, max_epochs=tcfg.max_epochs)
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
                v = train_ae(model, data, tcfg, patience=acfg.patience, max_epochs=tcfg.max_epochs)
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
                v = train_ae(model, data, tcfg, patience=acfg.patience, max_epochs=tcfg.max_epochs)
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
    p = argparse.ArgumentParser(description="ADP AE (graph fully-connected) width/depth search")
    p.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "Citeseer", "PubMed"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=32)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=16)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    args = p.parse_args()

    data, _ = load_planetoid(args.dataset)
    X, _, _, _, _ = data
    N = X.size(0)
    model = DNNNodeAE(N, hidden=args.hidden, depth=args.depth)
    tcfg = TrainCfg(patience=args.patience)
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
    )
    best = adp_search(model, data, tcfg, acfg)
    print(f\"[ADP AE] dataset={args.dataset} mode={args.adp_mode} best_val={best:.6f} hidden={model.in_lin.out_features} depth={len(model.hiddens)+1}\")


if __name__ == "__main__":
    main()
