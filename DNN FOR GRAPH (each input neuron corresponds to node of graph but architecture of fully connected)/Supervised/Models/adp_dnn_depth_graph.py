
from dataclasses import dataclass
import torch
from adp_dnn_core import AdaptiveDNNNodeFC, TrainCfg, train_early_stop, load_planetoid

@dataclass
class SearchCfg:
    ex_k: int = 8
    max_width: int = 2048
    max_neurons: int = 200_000_000
    delta: float = 1e-4
    trials_depth: int = 100
    trials_width: int = 100

def fit_depth_then_width(dataset: str = "Cora", init_hidden: int = 64, init_depth: int = 2, seed: int = 42):
    torch.manual_seed(seed)
    data, num_classes = load_planetoid(dataset)
    X, y, train_mask, val_mask, test_mask = data
    num_nodes = X.size(0)
    model = AdaptiveDNNNodeFC(num_nodes=num_nodes, num_classes=num_classes, hidden=init_hidden, depth=init_depth)
    tcfg = TrainCfg(); scfg = SearchCfg()
    best_val, _, _ = train_early_stop(model, data, tcfg)
    best_state = model.snapshot(); baseline = best_val
    depth_pat = scfg.trials_depth
    while depth_pat > 0:
        proj = model.hidden * (model.depth + 2) + model.num_nodes * model.num_classes
        if proj > scfg.max_neurons: break
        model.append_depth()
        val, _, _ = train_early_stop(model, data, tcfg)
        if val < baseline - scfg.delta:
            baseline = val; best_state = model.snapshot()
            width_pat = scfg.trials_width
            while width_pat > 0:
                proj = (model.hidden + scfg.ex_k) * (model.depth + 1) + model.num_nodes * model.num_classes
                if proj > scfg.max_neurons or (model.hidden + scfg.ex_k) > scfg.max_width: break
                model.widen_all(scfg.ex_k)
                v2, _, _ = train_early_stop(model, data, tcfg)
                if v2 < baseline - scfg.delta:
                    baseline = v2; best_state = model.snapshot()
                else:
                    model.restore(best_state); width_pat -= 1
        else:
            model.hiddens = model.hiddens[:-1]; model.depth -= 1; depth_pat -= 1
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
    model, v, t = fit_depth_then_width(args.dataset, args.init_hidden, args.init_depth)
    print(f"[ADP-Depth→Width] {args.dataset} final Val={v:.4f} Test@1={t*100:.2f}%  H={model.hidden} D={model.depth}")

if __name__ == "__main__":
    main()
