
from dataclasses import dataclass
import torch
from adp_dnn_core import AdaptiveDNNNodeFC, TrainCfg, train_early_stop, load_planetoid

@dataclass
class AltCfg:
    ex_k: int = 4
    delta: float = 1e-4
    patience_depth: int = 20
    patience_width: int = 20
    max_neurons: int = 200_000_000
    max_width: int = 4096

def alternating_depth_first(dataset: str = "Cora", init_hidden: int = 64, init_depth: int = 2, seed: int = 42):
    torch.manual_seed(seed)
    data, num_classes = load_planetoid(dataset)
    X, y, train_mask, val_mask, test_mask = data
    N = X.size(0)
    model = AdaptiveDNNNodeFC(N, num_classes, hidden=init_hidden, depth=init_depth)
    tcfg = TrainCfg(); acfg = AltCfg()
    best_val, _, _ = train_early_stop(model, data, tcfg)
    best_state = model.snapshot(); baseline = best_val
    while True:
        improved = False
        dpat = acfg.patience_depth
        while dpat > 0:
            proj = model.hidden * (model.depth + 2) + model.num_nodes * model.num_classes
            if proj > acfg.max_neurons: break
            model.append_depth()
            v, _, _ = train_early_stop(model, data, tcfg)
            if v < baseline - acfg.delta:
                baseline = v; best_state = model.snapshot(); improved = True
            else:
                model.hiddens = model.hiddens[:-1]; model.depth -= 1; dpat -= 1
        wpat = acfg.patience_width
        while wpat > 0:
            proj = (model.hidden + acfg.ex_k) * (model.depth + 1) + model.num_nodes * model.num_classes
            if proj > acfg.max_neurons or (model.hidden + acfg.ex_k) > acfg.max_width: break
            model.widen_all(acfg.ex_k)
            v, _, _ = train_early_stop(model, data, tcfg)
            if v < baseline - acfg.delta:
                baseline = v; best_state = model.snapshot(); improved = True
            else:
                model.restore(best_state); wpat -= 1
        if not improved: break
    model.restore(best_state)
    final_val, final_test, _ = train_early_stop(model, data, tcfg)
    return model, final_val, final_test

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Cora", choices=["Cora","Citeseer","PubMed"])
    p.add_argument("--init-hidden", type=int, default=64)
    p.add_argument("--init-depth", type=int, default=2)
    args = p.parse_args()
    model, v, t = alternating_depth_first(args.dataset, args.init_hidden, args.init_depth)
    print(f"[ADP-Alt (Depth-first)] {args.dataset} final Val={v:.4f} Test@1={t*100:.2f}%  H={model.hidden} D={model.depth}")

if __name__ == "__main__":
    main()
