import copy
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from DNN_FOR_GRAPH_each_input_neuron_corresponds_to_node_of_graph_but_architecture_of_fully_connected.Supervised.Models.dnn_stl_graph import (  # noqa: E501
    DNNNodeFC,
    TrainCfg as BaseTrainCfg,
    load_planetoid,
)


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"  # {"width_only","depth_only","width_to_depth","depth_to_width","alt_width","alt_depth","width","depth"}
    delta: float = 1e-3               # improvement margin
    patience: int = 50                # inner early-stopping patience
    trials_width: int = 2             # failed width expansions before stop/rollback
    trials_depth: int = 2             # failed depth expansions before stop/rollback
    ex_k: int = 32                    # width increment
    max_width: int = 2048
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


def widen_all(model: DNNNodeFC, ex_k: int, max_width: int):
    """Increase hidden width of all layers by ex_k (capped)."""
    # in_lin
    new_h = min(max_width, model.in_lin.out_features + ex_k)
    model.in_lin = _resize_linear(model.in_lin, new_h, model.in_lin.in_features)
    prev_out = new_h
    new_hiddens = nn.ModuleList()
    for lin in model.hiddens:
        nh = min(max_width, lin.out_features + ex_k)
        new_hiddens.append(_resize_linear(lin, nh, prev_out))
        prev_out = nh
    model.hiddens = new_hiddens
    model.out_lin = _resize_linear(model.out_lin, model.out_lin.out_features, prev_out)


def append_depth(model: DNNNodeFC):
    """Insert one hidden layer (square, width=last hidden) before out_lin."""
    width = model.hiddens[-1].out_features if len(model.hiddens) > 0 else model.in_lin.out_features
    device = model.in_lin.weight.device
    model.hiddens.append(nn.Linear(width, width, bias=False).to(device))
    model.out_lin = _resize_linear(model.out_lin, model.out_lin.out_features, width)


def total_neurons(model: DNNNodeFC) -> int:
    h = model.in_lin.out_features
    return h * (len(model.hiddens) + 1) + model.out_lin.in_features * model.out_lin.out_features


def train_with_patience(model: DNNNodeFC, data, cfg: BaseTrainCfg, patience: int, max_epochs: int) -> float:
    X, y, train_mask, val_mask, _ = data
    device = cfg.device
    X = X.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf")
    best_state = None
    pat = patience
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        # val
        model.eval()
        with torch.no_grad():
            val = F.cross_entropy(model(X)[val_mask], y[val_mask]).item()
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


def adp_search(model: DNNNodeFC, data, tcfg: BaseTrainCfg, acfg: ADPConfig):
    """Unified ADP search handling width/depth/alt policies."""
    def can_widen() -> bool:
        return model.in_lin.out_features + acfg.ex_k <= acfg.max_width and total_neurons(model) < acfg.max_neurons

    def can_deepen() -> bool:
        return len(model.hiddens) + 1 <= acfg.max_depth and (total_neurons(model) + model.in_lin.out_features) <= acfg.max_neurons

    def inner_train():
        return train_with_patience(model, data, tcfg, patience=acfg.patience, max_epochs=tcfg.max_epochs)

    def try_width():
        pre = copy.deepcopy(model.state_dict())
        pre_val = inner_val
        widen_all(model, acfg.ex_k, acfg.max_width)
        v = inner_train()
        return v < pre_val - acfg.delta, v, pre

    def try_depth():
        pre = copy.deepcopy(model.state_dict())
        pre_val = inner_val
        append_depth(model)
        v = inner_train()
        return v < pre_val - acfg.delta, v, pre

    inner_val = inner_train()
    best_state = copy.deepcopy(model.state_dict())
    best_val = inner_val

    pw = acfg.trials_width
    pd = acfg.trials_depth

    mode = acfg.adp_mode
    improved = True
    while improved:
        improved = False
        if mode in ("width_only", "width", "width_to_depth", "alt_width"):
            if can_widen() and pw > 0:
                ok, v, pre = try_width()
                if ok:
                    inner_val = v
                    improved = True
                    pw = acfg.trials_width
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
                ok, v, pre = try_depth()
                if ok:
                    inner_val = v
                    improved = True
                    pd = acfg.trials_depth
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
    p = argparse.ArgumentParser(description="ADP DNN Graph (width/depth search)")
    p.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "Citeseer", "PubMed"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=32)
    p.add_argument("--max-width", type=int, default=2048)
    p.add_argument("--max-depth", type=int, default=16)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    args = p.parse_args()

    data, num_classes = load_planetoid(args.dataset)
    X, _, _, _, _ = data
    N = X.size(0)
    model = DNNNodeFC(N, num_classes, hidden=args.hidden, depth=args.depth)
    tcfg = BaseTrainCfg(patience=args.patience)
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
    print(f"[ADP] dataset={args.dataset} mode={args.adp_mode} best_val={best:.6f} hidden={model.in_lin.out_features} depth={len(model.hiddens)+1}")


if __name__ == "__main__":
    main()
