"""
Generic ADP (Adaptive Depth/Width) search loops shared by DAE runners.
Policies mirror the ones used in other ADP modules: width_only, depth_only,
width_to_depth, depth_to_width, alt_width_first, alt_depth_first.
"""
from dataclasses import dataclass
from typing import Callable, Any

import torch


@dataclass
class SearchConfig:
    delta: float = 1e-3          # improvement threshold
    patience_width: int = 2
    patience_depth: int = 2
    ex_k: int = 16               # width expansion
    max_neurons: int = 2_000_000
    max_depth: int = 32
    max_width: int = 1024
    max_total_epochs: int | None = None


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore(model, snap):
    model.load_state_dict(snap)


def can_widen(model, ex_k: int, scfg: SearchConfig) -> bool:
    if ex_k <= 0:
        return False
    projected = model.total_neurons() + ex_k * model.depth()
    if projected > scfg.max_neurons:
        return False
    if any(w + ex_k > scfg.max_width for w in model.widths_list()):
        return False
    return True


def can_deepen(model, scfg: SearchConfig) -> bool:
    if model.depth() + 1 > scfg.max_depth:
        return False
    projected = model.total_neurons() + 2 * model.widths_list()[-1]
    return projected <= scfg.max_neurons


def _train_eval(model, train_fn: Callable[[], float], val_fn: Callable[[], float], max_epochs: int) -> float:
    # train_fn runs one epoch over loader; val_fn computes validation metric (loss)
    for _ in range(max_epochs):
        train_fn()
    return val_fn()


def _policy_loop(model, scfg: SearchConfig, max_epochs: int, train_fn: Callable[[], float], val_fn: Callable[[], float],
                 first: str):
    best_snap = snapshot(model)
    best_val = _train_eval(model, train_fn, val_fn, max_epochs)
    total = 0
    ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs

    def width_phase():
        nonlocal best_val, best_snap, total
        fails = 0
        while fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(model)
            model.widen_all(scfg.ex_k)
            v = _train_eval(model, train_fn, val_fn, max_epochs)
            total += max_epochs
            if v < best_val - scfg.delta:
                best_val = v
                best_snap = snapshot(model)
                fails = 0
            else:
                fails += 1
                restore(model, pre)

    def depth_phase():
        nonlocal best_val, best_snap, total
        fails = 0
        while fails < scfg.patience_depth and can_deepen(model, scfg) and ok(total):
            pre = snapshot(model)
            model.append_depth()
            v = _train_eval(model, train_fn, val_fn, max_epochs)
            total += max_epochs
            if v < best_val - scfg.delta:
                best_val = v
                best_snap = snapshot(model)
                fails = 0
            else:
                fails += 1
                restore(model, pre)

    improved = True
    while improved and ok(total):
        improved = False
        if first == "width":
            width_phase()
            depth_phase()
        else:
            depth_phase()
            width_phase()
        # check if we actually improved over last outer loop
        curr_val = _train_eval(model, train_fn, val_fn, 0)
        if curr_val < best_val - scfg.delta:
            best_val = curr_val
            best_snap = snapshot(model)
            improved = True
    restore(model, best_snap)
    return model


def search_width_only(model, scfg: SearchConfig, max_epochs: int, train_fn, val_fn):
    best_snap = snapshot(model)
    best_val = _train_eval(model, train_fn, val_fn, max_epochs)
    fails = 0
    while fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg):
        pre = snapshot(model)
        model.widen_all(scfg.ex_k)
        v = _train_eval(model, train_fn, val_fn, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v
            best_snap = snapshot(model)
        else:
            fails += 1
            restore(model, pre)
    restore(model, best_snap)
    return model


def search_depth_only(model, scfg: SearchConfig, max_epochs: int, train_fn, val_fn):
    best_snap = snapshot(model)
    best_val = _train_eval(model, train_fn, val_fn, max_epochs)
    fails = 0
    while fails < scfg.patience_depth and can_deepen(model, scfg):
        pre = snapshot(model)
        model.append_depth()
        v = _train_eval(model, train_fn, val_fn, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v
            best_snap = snapshot(model)
        else:
            fails += 1
            restore(model, pre)
    restore(model, best_snap)
    return model


def search_width_to_depth(model, scfg: SearchConfig, max_epochs: int, train_fn, val_fn):
    # width steps outer, depth inner
    best_snap = snapshot(model)
    best_val = _train_eval(model, train_fn, val_fn, max_epochs)
    width_fails = 0
    while width_fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg):
        pre = snapshot(model)
        model.widen_all(scfg.ex_k)
        v = _train_eval(model, train_fn, val_fn, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v
            best_snap = snapshot(model)
            depth_fails = 0
            while depth_fails < scfg.patience_depth and can_deepen(model, scfg):
                pre2 = snapshot(model)
                model.append_depth()
                v2 = _train_eval(model, train_fn, val_fn, max_epochs)
                if v2 < best_val - scfg.delta:
                    best_val = v2
                    best_snap = snapshot(model)
                else:
                    depth_fails += 1
                    restore(model, pre2)
        else:
            width_fails += 1
            restore(model, pre)
    restore(model, best_snap)
    return model


def search_depth_to_width(model, scfg: SearchConfig, max_epochs: int, train_fn, val_fn):
    # depth steps outer, width inner
    best_snap = snapshot(model)
    best_val = _train_eval(model, train_fn, val_fn, max_epochs)
    depth_fails = 0
    while depth_fails < scfg.patience_depth and can_deepen(model, scfg):
        pre = snapshot(model)
        model.append_depth()
        v = _train_eval(model, train_fn, val_fn, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v
            best_snap = snapshot(model)
            width_fails = 0
            while width_fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg):
                pre2 = snapshot(model)
                model.widen_all(scfg.ex_k)
                v2 = _train_eval(model, train_fn, val_fn, max_epochs)
                if v2 < best_val - scfg.delta:
                    best_val = v2
                    best_snap = snapshot(model)
                else:
                    width_fails += 1
                    restore(model, pre2)
        else:
            depth_fails += 1
            restore(model, pre)
    restore(model, best_snap)
    return model


def search_alt_width_first(model, scfg: SearchConfig, max_epochs: int, train_fn, val_fn):
    return _policy_loop(model, scfg, max_epochs, train_fn, val_fn, first="width")


def search_alt_depth_first(model, scfg: SearchConfig, max_epochs: int, train_fn, val_fn):
    return _policy_loop(model, scfg, max_epochs, train_fn, val_fn, first="depth")
