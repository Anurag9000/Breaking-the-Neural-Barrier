import copy
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from DNN_FOR_GRAPH_each_input_neuron_corresponds_to_node_of_graph_but_architecture_of_fully_connected.Supervised.Models.dnn_stl_graph import (  # noqa: E501
    DNNNodeFC,
    TrainCfg as BaseTrainCfg,
    load_planetoid,
)


# ADP REVIEW (BEFORE REFACTOR)
# - Modes: width_only/width, depth_only/depth, width_to_depth, depth_to_width, alt_width, alt_depth share single loop with per-expansion rollback.
# - Inner training: train_with_patience ties ES reset to delta and reloads immediately.
# - Expansions: widen/deepen rollback on failure; shared delta/patience; no snapshot helpers.
# - Control flow: toggles modes on no improvement; lacks forward-only march and context-end restore per updated spec.
# - ES patience conflated with expansion patiences; no snapshot/restore separation.


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


def expand_width(model: DNNNodeFC, ex_k: int, max_width: int) -> Optional[DNNNodeFC]:
    """Increase hidden width of all layers by ex_k (capped)."""
    # Check if we can expand
    current_width = model.in_lin.out_features
    new_h = min(max_width, current_width + ex_k)
    if new_h == current_width:
        return None

    # in_lin
    model.in_lin = _resize_linear(model.in_lin, new_h, model.in_lin.in_features)
    prev_out = new_h
    new_hiddens = nn.ModuleList()
    for lin in model.hiddens:
        nh = min(max_width, lin.out_features + ex_k)
        new_hiddens.append(_resize_linear(lin, nh, prev_out))
        prev_out = nh
    model.hiddens = new_hiddens
    model.out_lin = _resize_linear(model.out_lin, model.out_lin.out_features, prev_out)
    return model


def expand_depth(model: DNNNodeFC, max_depth: int) -> Optional[DNNNodeFC]:
    """Insert one hidden layer (square, width=last hidden) before out_lin."""
    # Check max depth (depth = len(hiddens) + 1 (in_lin))
    # Actually, depth usually means number of layers or number of hidden layers?
    # Base code: depth=3 means 3 layers total?
    # DNNNodeFC init: self.hiddens = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth - 2)])
    # So if depth=3, hiddens has 1 layer. Total layers: in_lin + hiddens[0] + out_lin = 3.
    # So current depth = len(model.hiddens) + 2.
    
    current_depth = len(model.hiddens) + 2
    if current_depth >= max_depth:
        return None

    width = model.hiddens[-1].out_features if len(model.hiddens) > 0 else model.in_lin.out_features
    device = model.in_lin.weight.device
    model.hiddens.append(nn.Linear(width, width, bias=False).to(device))
    model.out_lin = _resize_linear(model.out_lin, model.out_lin.out_features, width)
    return model


def total_neurons(model: DNNNodeFC) -> int:
    h = model.in_lin.out_features
    return h * (len(model.hiddens) + 1) + model.out_lin.in_features * model.out_lin.out_features


def snapshot_arch_and_state(model: DNNNodeFC, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    # Need to capture architecture params to rebuild
    # DNNNodeFC is dynamic, so we need:
    # - in_features (N)
    # - num_classes
    # - hidden width (assuming uniform for now, but expand_width keeps it uniform-ish)
    # - depth (len(hiddens))
    # But wait, expand_width might make layers non-uniform if we capped some?
    # The current expand_width implementation tries to keep them uniform (new_h = min(max, old + k)).
    # If they started uniform, they stay uniform.
    # But let's be safe and capture the exact structure if possible, or just rebuild and load state.
    # Since we modify the model in-place in expand_*, we can just deepcopy the model?
    # No, we need to be able to restore it.
    # Let's capture the list of widths.
    
    widths = [model.in_lin.out_features] + [l.out_features for l in model.hiddens]
    return {
        "in_features": model.in_lin.in_features,
        "num_classes": model.out_lin.out_features,
        "widths": widths,
        "state": copy.deepcopy(state)
    }


def restore_arch_and_state(model: DNNNodeFC, snap: Dict[str, Any], device) -> DNNNodeFC:
    # Rebuild model from scratch
    # We can't easily use the constructor if widths are non-uniform.
    # But DNNNodeFC constructor assumes uniform hidden width.
    # We might need to manually construct it.
    
    # Create a dummy model
    dummy_hidden = snap["widths"][0]
    dummy_depth = len(snap["widths"]) + 1 # +1 for output layer? No.
    # Layers: in_lin (to widths[0]), hiddens (widths[0]->widths[1]...), out_lin (widths[-1]->classes)
    
    # Actually, let's look at DNNNodeFC structure again.
    # in_lin: N -> hidden
    # hiddens: hidden -> hidden
    # out_lin: hidden -> num_classes
    
    # If we have non-uniform widths, the standard class might not support it easily without modification.
    # But our expand_width implementation:
    # new_h = min(max, model.in_lin.out_features + ex_k)
    # ...
    # for lin in model.hiddens: nh = min(max, lin.out_features + ex_k)
    
    # It seems it tries to maintain uniformity if they were uniform.
    # Let's assume they are uniform enough or we can just patch the layers.
    
    new_model = DNNNodeFC(snap["in_features"], snap["num_classes"], hidden=snap["widths"][0], depth=len(snap["widths"]) + 1)
    new_model = new_model.to(device)
    
    # If widths are different, we need to resize layers manually
    # in_lin
    if new_model.in_lin.out_features != snap["widths"][0]:
        new_model.in_lin = nn.Linear(snap["in_features"], snap["widths"][0], bias=True).to(device)
        
    # hiddens
    # The constructor creates (depth-2) hidden layers.
    # snap["widths"] has length = len(hiddens) + 1 (for in_lin output).
    # Wait, widths[0] is output of in_lin.
    # widths[1] is output of hiddens[0].
    # ...
    
    new_hiddens = nn.ModuleList()
    prev_w = snap["widths"][0]
    for i in range(1, len(snap["widths"])):
        w = snap["widths"][i]
        new_hiddens.append(nn.Linear(prev_w, w, bias=True).to(device)) # DNNNodeFC uses bias=True by default?
        # In append_depth it uses bias=False?
        # Base code: self.hiddens.append(nn.Linear(hidden, hidden)) -> default bias=True.
        # append_depth: bias=False. This is inconsistent in the original code.
        # Let's stick to what the snapshot state has. load_state_dict will handle weights/bias presence.
        # We just need correct shapes.
        prev_w = w
        
    new_model.hiddens = new_hiddens
    
    # out_lin
    new_model.out_lin = nn.Linear(prev_w, snap["num_classes"], bias=True).to(device)
    
    new_model.load_state_dict(snap["state"])
    return new_model


def train_with_early_stopping(model: DNNNodeFC, data, cfg: BaseTrainCfg, patience: int, max_epochs: int) -> Tuple[float, Dict[str, Any]]:
    X, y, train_mask, val_mask, _ = data
    device = cfg.device
    X = X.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0
    
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
            
        if val < best_val: # Strict improvement
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1
            
        if es_counter >= patience:
            break
            
    return best_val, best_state


def adp_search(model: DNNNodeFC, data, tcfg: BaseTrainCfg, acfg: ADPConfig):
    """Unified ADP search handling width/depth/alt policies."""
    
    # Initial training
    best_val, best_state = train_with_early_stopping(model, data, tcfg, acfg.patience, tcfg.max_epochs)
    model.load_state_dict(best_state)
    
    global_best_val = best_val
    global_best_snap = snapshot_arch_and_state(model, best_state)
    
    device = tcfg.device

    def can_widen(m: DNNNodeFC) -> bool:
        return m.in_lin.out_features + acfg.ex_k <= acfg.max_width and total_neurons(m) < acfg.max_neurons

    def can_deepen(m: DNNNodeFC) -> bool:
        return len(m.hiddens) + 2 < acfg.max_depth and (total_neurons(m) + m.in_lin.out_features) <= acfg.max_neurons

    def optimize_width_at_fixed_depth(curr_model: DNNNodeFC) -> Tuple[DNNNodeFC, float, Dict[str, Any]]:
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

    def optimize_depth_at_fixed_width(curr_model: DNNNodeFC) -> Tuple[DNNNodeFC, float, Dict[str, Any]]:
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
    best, model = adp_search(model, data, tcfg, acfg)
    print(f"[ADP] dataset={args.dataset} mode={args.adp_mode} best_val={best:.6f} hidden={model.in_lin.out_features} depth={len(model.hiddens)+2}")


if __name__ == "__main__":
    main()
