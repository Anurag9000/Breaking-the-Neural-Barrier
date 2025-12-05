import copy
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

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


# ADP REVIEW (BEFORE REFACTOR)
# - Modes: width_only/width, depth_only/depth, width_to_depth, depth_to_width, alt_width, alt_depth share single loop with per-expansion rollback.
# - Inner training: train_with_patience ties ES reset to delta and reloads immediately.
# - Expansions: widen/deepen rollback on failure; shared delta/patience; no snapshot helpers.
# - Control flow: toggles modes on no improvement; lacks forward-only march and context-end restore per updated spec.
# - ES patience conflated with expansion patiences; no snapshot/restore separation.


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


def expand_width(model: DNNNodeAE, ex_k: int, max_width: int) -> Optional[DNNNodeAE]:
    """Increase hidden width everywhere by ex_k (capped)."""
    current_width = model.in_lin.out_features
    new_h = min(max_width, current_width + ex_k)
    if new_h == current_width:
        return None

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
    return model


def expand_depth(model: DNNNodeAE, max_depth: int) -> Optional[DNNNodeAE]:
    """Append one hidden layer (square) before decoder_out."""
    # Current depth = len(hiddens) + 1 (in_lin)
    # Actually, DNNNodeAE depth usually refers to number of layers.
    # If depth=3, hiddens has 1 layer.
    
    current_depth = len(model.hiddens) + 2 # in_lin + hiddens + decoder
    if current_depth >= max_depth:
        return None

    width = model.hidden
    device = model.in_lin.weight.device
    model.hiddens.append(nn.Linear(width, width, bias=False).to(device))
    if model.decoder_out is not None:
        model.decoder_out = _resize_linear(model.decoder_out, model.decoder_out.out_features, width)
    model.depth = len(model.hiddens) + 1
    return model


def snapshot_arch_and_state(model: DNNNodeAE, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    # Capture widths
    widths = [model.in_lin.out_features] + [l.out_features for l in model.hiddens]
    return {
        "in_features": model.in_lin.in_features,
        "widths": widths,
        "state": copy.deepcopy(state)
    }


def restore_arch_and_state(model: DNNNodeAE, snap: Dict[str, Any], device) -> DNNNodeAE:
    # Rebuild
    # DNNNodeAE(in_dim, hidden, depth)
    # depth = len(widths) + 1?
    # in_lin -> widths[0]
    # hiddens -> widths[1]...
    
    new_model = DNNNodeAE(snap["in_features"], hidden=snap["widths"][0], depth=len(snap["widths"]) + 1)
    new_model = new_model.to(device)
    
    # Resize if needed
    if new_model.in_lin.out_features != snap["widths"][0]:
        new_model.in_lin = nn.Linear(snap["in_features"], snap["widths"][0], bias=True).to(device)
        
    new_hiddens = nn.ModuleList()
    prev_w = snap["widths"][0]
    for i in range(1, len(snap["widths"])):
        w = snap["widths"][i]
        new_hiddens.append(nn.Linear(prev_w, w, bias=True).to(device)) # Assuming bias=True for base layers
        prev_w = w
    new_model.hiddens = new_hiddens
    
    # decoder_out is created lazily in forward usually, but if we load state dict it might expect it?
    # DNNNodeAE forward: if self.decoder_out is None: self.decoder_out = nn.Linear(self.hidden, N)
    # So we don't need to create it here unless it's in the state dict.
    # But wait, if we load state dict, we need the parameter to exist.
    # Let's check if 'decoder_out.weight' is in snap["state"].
    if "decoder_out.weight" in snap["state"]:
        # We need to know output dim. It's in_features (AE).
        new_model.decoder_out = nn.Linear(prev_w, snap["in_features"], bias=True).to(device)
    
    new_model.hidden = prev_w
    new_model.depth = len(snap["widths"]) + 1
    
    new_model.load_state_dict(snap["state"])
    return new_model


def train_with_early_stopping(model: DNNNodeAE, data, cfg: TrainCfg, patience: int, max_epochs: int) -> Tuple[float, Dict[str, Any]]:
    X, _, train_mask, val_mask, _ = data
    X = X.to(cfg.device)
    train_mask = train_mask.to(cfg.device)
    val_mask = val_mask.to(cfg.device)
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0
    
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
            
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1
            
        if es_counter >= patience:
            break
            
    return best_val, best_state


def adp_search(model: DNNNodeAE, data, tcfg: TrainCfg, acfg: ADPConfig):
    """Unified ADP search across width/depth policies."""
    
    # Initial training
    best_val, best_state = train_with_early_stopping(model, data, tcfg, acfg.patience, tcfg.max_epochs)
    model.load_state_dict(best_state)
    
    global_best_val = best_val
    global_best_snap = snapshot_arch_and_state(model, best_state)
    
    device = tcfg.device

    def can_widen(m: DNNNodeAE) -> bool:
        return m.in_lin.out_features + acfg.ex_k <= acfg.max_width and total_neurons(m) < acfg.max_neurons

    def can_deepen(m: DNNNodeAE) -> bool:
        return (len(m.hiddens) + 2) < acfg.max_depth and (total_neurons(m) + m.hidden) <= acfg.max_neurons

    def optimize_width_at_fixed_depth(curr_model: DNNNodeAE) -> Tuple[DNNNodeAE, float, Dict[str, Any]]:
        local_val, local_state = train_with_early_stopping(curr_model, data, tcfg, acfg.patience, tcfg.max_epochs)
        local_best_val = local_val
        local_best_state = local_state
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        
        width_failure_count = 0
        
        while width_failure_count < acfg.trials_width:
            if not can_widen(curr_model):
                break
                
            next_model = expand_width(curr_model, acfg.ex_k, acfg.max_width)
            if next_model is None:
                break
            curr_model = next_model
            
            v, s = train_with_early_stopping(curr_model, data, tcfg, acfg.patience, tcfg.max_epochs)
            
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_state = s
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                width_failure_count = 0
            else:
                width_failure_count += 1
        
        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        return final_model, local_best_val, local_best_snap

    def optimize_depth_at_fixed_width(curr_model: DNNNodeAE) -> Tuple[DNNNodeAE, float, Dict[str, Any]]:
        local_val, local_state = train_with_early_stopping(curr_model, data, tcfg, acfg.patience, tcfg.max_epochs)
        local_best_val = local_val
        local_best_state = local_state
        local_best_snap = snapshot_arch_and_state(curr_model, local_state)
        
        depth_failure_count = 0
        
        while depth_failure_count < acfg.trials_depth:
            if not can_deepen(curr_model):
                break
                
            next_model = expand_depth(curr_model, acfg.max_depth)
            if next_model is None:
                break
            curr_model = next_model
            
            v, s = train_with_early_stopping(curr_model, data, tcfg, acfg.patience, tcfg.max_epochs)
            
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_state = s
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                depth_failure_count = 0
            else:
                depth_failure_count += 1
        
        final_model = restore_arch_and_state(curr_model, local_best_snap, device)
        return final_model, local_best_val, local_best_snap

    mode = acfg.adp_mode
    
    if mode in ["width_only", "width"]:
        model, global_best_val, global_best_snap = optimize_width_at_fixed_depth(model)
        
    elif mode in ["depth_only", "depth"]:
        model, global_best_val, global_best_snap = optimize_depth_at_fixed_width(model)
        
    elif mode == "depth_to_width":
        model, base_val, base_snap = optimize_width_at_fixed_depth(model)
        global_best_val = base_val
        global_best_snap = base_snap
        
        depth_failure_count = 0
        while depth_failure_count < acfg.trials_depth and len(model.hiddens) + 2 < acfg.max_depth:
            if not can_deepen(model):
                break
            
            next_model = expand_depth(model, acfg.max_depth)
            if next_model is None:
                break
            model = next_model
            
            model, val_d, snap_d = optimize_width_at_fixed_depth(model)
            
            if val_d < global_best_val - acfg.delta:
                global_best_val = val_d
                global_best_snap = snap_d
                depth_failure_count = 0
            else:
                depth_failure_count += 1
        
        model = restore_arch_and_state(model, global_best_snap, device)

    elif mode == "width_to_depth":
        model, base_val, base_snap = optimize_depth_at_fixed_width(model)
        global_best_val = base_val
        global_best_snap = base_snap
        
        width_failure_count = 0
        while width_failure_count < acfg.trials_width and model.in_lin.out_features < acfg.max_width:
            if not can_widen(model):
                break
            
            next_model = expand_width(model, acfg.ex_k, acfg.max_width)
            if next_model is None:
                break
            model = next_model
            
            model, val_w, snap_w = optimize_depth_at_fixed_width(model)
            
            if val_w < global_best_val - acfg.delta:
                global_best_val = val_w
                global_best_snap = snap_w
                width_failure_count = 0
            else:
                width_failure_count += 1
        
        model = restore_arch_and_state(model, global_best_snap, device)

    elif mode in ["alt_width", "alt_depth"]:
        depth_saturated = False
        width_saturated = False
        current_phase = "width" if mode == "alt_width" else "depth"
        
        while not (depth_saturated and width_saturated):
            improved_in_phase = False
            
            if current_phase == "width":
                model, val, snap = optimize_width_at_fixed_depth(model)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved_in_phase = True
                
                width_saturated = not improved_in_phase
                model = restore_arch_and_state(model, global_best_snap, device)
                current_phase = "depth"
            else:
                model, val, snap = optimize_depth_at_fixed_width(model)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved_in_phase = True
                
                depth_saturated = not improved_in_phase
                model = restore_arch_and_state(model, global_best_snap, device)
                current_phase = "width"
                
        model = restore_arch_and_state(model, global_best_snap, device)

    # ADP REVIEW (AFTER REFACTOR)
    # - Implemented forward-only logic for all modes.
    # - optimize_width_at_fixed_depth / optimize_depth_at_fixed_width helpers.
    # - Global best restoration at end of contexts.
    # - train_with_early_stopping: ES counter only.
    # - snapshot/restore: Captures full architecture (widths list).

    return global_best_val, model


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
    best, model = adp_search(model, data, tcfg, acfg)
    print(f"[ADP AE] dataset={args.dataset} mode={args.adp_mode} best_val={best:.6f} hidden={model.in_lin.out_features} depth={len(model.hiddens)+1}")


if __name__ == "__main__":
    main()
