
import torch
from adp_nlp_ssl_common import AdaptiveTextSSL

def train_inner(model, trl, val, device, epochs, lr, patience, temperature):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.global_epoch += 1
        model.train(); tr_sum, tr_n = 0.0, 0
        for (v1, v2) in trl:
            (i1, l1), (i2, l2) = v1, v2
            i1, l1, i2, l2 = i1.to(device), l1.to(device), i2.to(device), l2.to(device)
            loss = model((i1,l1), (i2,l2), temperature=temperature)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item() * i1.size(0); tr_n += i1.size(0)

        model.eval(); va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for (v1, v2) in val:
                (i1, l1), (i2, l2) = v1, v2
                i1, l1, i2, l2 = i1.to(device), l1.to(device), i2.to(device), l2.to(device)
                loss = model((i1,l1), (i2,l2), temperature=temperature)
                va_sum += loss.item() * i1.size(0); va_n += i1.size(0)
        va = va_sum / max(va_n, 1)

        if va < best_val:
            best_val, best_state, bad = va, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        print(f"[{model.global_epoch:04d}] val_ssl={va:.6f} | depth={model.depth()} neurons={model.total_neurons()} widths={model.hidden}+[rep={model.rep_dim},proj={model.proj_dim}]")
        if bad >= patience: break
    return best_val, best_state

def adp_search_width_only(model, trl, val, device,
                          trials_width, epochs, lr, patience, delta, ex_k, temperature,
                          max_neurons=None, max_depth=None, max_width=None):
    best_val, best_state = train_inner(model, trl, val, device, epochs, lr, patience, temperature)
    print(f"Initial val_ssl={best_val:.6f}")
    w_trials = 0
    while w_trials < trials_width:
        if max_neurons is not None and model.total_neurons() >= max_neurons: break
        snap = model.snapshot(); model.widen_all(ex_k=ex_k)
        if max_width is not None and max(model.hidden+[model.rep_dim, model.proj_dim]) > max_width: model.restore(snap); break
        if max_depth is not None and model.depth() > max_depth: model.restore(snap); break
        val_loss, state = train_inner(model, trl, val, device, epochs, lr, patience, temperature)
        if val_loss + delta < best_val:
            print(f"ACCEPT width++ | {best_val:.6f} -> {val_loss:.6f}"); best_val, best_state = val_loss, state
        else:
            print(f"REJECT width++ | {val_loss:.6f}"); model.restore(snap)
        w_trials += 1
    if best_state is not None: model.load_state_dict(best_state, strict=True)
    return best_val
