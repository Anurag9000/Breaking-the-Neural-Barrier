# adp_mlp_ae_alt_width.py
# Width-first alternating search for AdaptiveMLPAE (single-model).
from typing import Optional
import torch

from adp_mlp_ae_alt_depth import AdaptiveMLPAE  # reuse your model

@torch.no_grad()
def _maybe_load_best(model: AdaptiveMLPAE, best_state):
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

def adp_search_alternating_width_first(
    model: AdaptiveMLPAE,
    train_loader,
    val_loader,
    device,
    cycles: int,
    d_steps: int,
    w_steps: int,
    epochs: int,
    lr: float,
    patience: int,
    delta: float,
    ex_k: int,
    max_neurons: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_width: Optional[int] = None,
    denoise_std: float = 0.0,
):
    # Initial fit
    best_val, best_state = model.train_inner(
        train_loader, val_loader, device, epochs, lr, patience, denoise_std
    )
    print(f"Initial val_mse={best_val:.6f}")

    for cy in range(1, cycles + 1):
        print(f"=== CYCLE {cy}/{cycles} : width-first ===")

        # ---- WIDTH STEPS (first) ----
        for _ in range(w_steps):
            if max_neurons is not None and model.total_neurons() >= max_neurons:
                print("Hit max_neurons; break width loop.")
                break

            snap = model.snapshot()
            model.widen_all(ex_k=ex_k)

            # guards
            if max_width is not None and max(model.hidden_widths + [model.bottleneck]) > max_width:
                print("Width step would exceed max_width; restoring.")
                model.restore(snap)
                break
            if max_depth is not None and model.depth() > max_depth:
                print("Width step invalidated depth guard; restoring.")
                model.restore(snap)
                break

            val, state = model.train_inner(
                train_loader, val_loader, device, epochs, lr, patience, denoise_std
            )
            if val + delta < best_val:
                print(f"ACCEPT width++ | {best_val:.6f} -> {val:.6f}")
                best_val, best_state = val, state
            else:
                print(f"REJECT width++ | {val:.6f} (>= {best_val:.6f} - delta)")
                model.restore(snap)

        _maybe_load_best(model, best_state)

        # ---- DEPTH STEPS (second) ----
        for _ in range(d_steps):
            if max_depth is not None and model.depth() >= max_depth:
                print("Hit max_depth; break depth loop.")
                break
            if max_neurons is not None and model.total_neurons() >= max_neurons:
                print("Hit max_neurons; break depth loop.")
                break

            snap = model.snapshot()
            model.append_depth()

            if max_width is not None and max(model.hidden_widths + [model.bottleneck]) > max_width:
                print("Depth step would exceed max_width; restoring.")
                model.restore(snap)
                break

            val, state = model.train_inner(
                train_loader, val_loader, device, epochs, lr, patience, denoise_std
            )
            if val + delta < best_val:
                print(f"ACCEPT depth++ | {best_val:.6f} -> {val:.6f}")
                best_val, best_state = val, state
            else:
                print(f"REJECT depth++ | {val:.6f} (>= {best_val:.6f} - delta)")
                model.restore(snap)

        _maybe_load_best(model, best_state)

    _maybe_load_best(model, best_state)
    return best_val
