
import torch
from adp_nlp_ae_common import AdaptiveTextAE
from nlp_ae_common import soft_ce_loss

def train_inner(model, trl, val, device, epochs, lr, patience):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.global_epoch += 1
        model.train(); tr_sum, tr_n = 0.0, 0
        for (tok, lens), bow in trl:
            tok, lens, bow = tok.to(device), lens.to(device), bow.to(device)
            loss = soft_ce_loss(model((tok, lens)), bow)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item() * tok.size(0); tr_n += tok.size(0)

        model.eval(); va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for (tok, lens), bow in val:
                tok, lens, bow = tok.to(device), lens.to(device), bow.to(device)
                loss = soft_ce_loss(model((tok, lens)), bow)
                va_sum += loss.item() * tok.size(0); va_n += tok.size(0)
        va = va_sum / max(va_n, 1)

        if va < best_val:
            best_val, best_state, bad = va, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        print(f"[{model.global_epoch:04d}] val_rec={va:.6f} | depth={model.depth()} neurons={model.total_neurons()} widths={model.hidden}+[rep={model.rep_dim}]")
        if bad >= patience: break
    return best_val, best_state

def adp_search_alternating_depth_first(model, trl, val, device,
                                       cycles, d_steps, w_steps, epochs, lr, patience,
                                       delta, ex_k, max_neurons=None, max_depth=None, max_width=None):
    best_val, best_state = train_inner(model, trl, val, device, epochs, lr, patience)
    print(f"Initial val_rec={best_val:.6f}")
    for cy in range(1, cycles+1):
        print(f"=== CYCLE {cy}/{cycles} : depth-first ===")
        for _ in range(d_steps):
            if max_depth is not None and model.depth() >= max_depth: break
            if max_neurons is not None and model.total_neurons() >= max_neurons: break
            snap = model.snapshot(); model.append_depth()
            if max_width is not None and max(model.hidden+[model.rep_dim]) > max_width: model.restore(snap); break
            val_loss, state = train_inner(model, trl, val, device, epochs, lr, patience)
            if val_loss + delta < best_val:
                print(f"ACCEPT depth++ | {best_val:.6f} -> {val_loss:.6f}"); best_val, best_state = val_loss, state
            else:
                print(f"REJECT depth++ | {val_loss:.6f}"); model.restore(snap)
        if best_state is not None: model.load_state_dict(best_state, strict=True)

        for _ in range(w_steps):
            if max_neurons is not None and model.total_neurons() >= max_neurons: break
            snap = model.snapshot(); model.widen_all(ex_k=ex_k)
            if max_width is not None and max(model.hidden+[model.rep_dim]) > max_width: model.restore(snap); break
            if max_depth is not None and model.depth() > max_depth: model.restore(snap); break
            val_loss, state = train_inner(model, trl, val, device, epochs, lr, patience)
            if val_loss + delta < best_val:
                print(f"ACCEPT width++ | {best_val:.6f} -> {val_loss:.6f}"); best_val, best_state = val_loss, state
            else:
                print(f"REJECT width++ | {val_loss:.6f}"); model.restore(snap)
        if best_state is not None: model.load_state_dict(best_state, strict=True)
    if best_state is not None: model.load_state_dict(best_state, strict=True)
    return best_val
