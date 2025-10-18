
from dataclasses import dataclass
import torch
from adp_dnn_core import AdaptiveDNNNodeFC, TrainCfg, train_early_stop, load_planetoid

@dataclass
class DepthOnlyCfg:
    delta: float = 1e-4
    trials_depth: int = 200
    max_neurons: int = 200_000_000

def fit_depth_only(dataset: str = "Cora", init_hidden: int = 64, init_depth: int = 1, seed: int = 42):
    torch.manual_seed(seed)
    data, num_classes = load_planetoid(dataset)
    X, y, train_mask, val_mask, test_mask = data
    N = X.size(0)
    model = AdaptiveDNNNodeFC(N, num_classes, hidden=init_hidden, depth=init_depth)
    tcfg = TrainCfg(); dcfg = DepthOnlyCfg()
    best_val, _, _ = train_early_stop(model, data, tcfg)
    best_state = model.snapshot(); baseline = best_val
    pat = dcfg.trials_depth
    while pat > 0:
        proj = model.hidden * (model.depth + 2) + model.num_nodes * model.num_classes
        if proj > dcfg.max_neurons: break
        model.append_depth()
        v, _, _ = train_early_stop(model, data, tcfg)
        if v < baseline - dcfg.delta:
            baseline = v; best_state = model.snapshot()
        else:
            model.hiddens = model.hiddens[:-1]; model.depth -= 1; pat -= 1
    model.restore(best_state)
    final_val, final_test, _ = train_early_stop(model, data, tcfg)
    return model, final_val, final_test

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Cora", choices=["Cora","Citeseer","PubMed"])
    p.add_argument("--init-hidden", type=int, default=64)
    p.add_argument("--init-depth", type=int, default=1)
    args = p.parse_args()
    model, v, t = fit_depth_only(args.dataset, args.init_hidden, args.init_depth)
    print(f"[ADP-Depth-Only] {args.dataset} final Val={v:.4f} Test@1={t*100:.2f}%  H={model.hidden} D={model.depth}")

if __name__ == "__main__":
    main()
