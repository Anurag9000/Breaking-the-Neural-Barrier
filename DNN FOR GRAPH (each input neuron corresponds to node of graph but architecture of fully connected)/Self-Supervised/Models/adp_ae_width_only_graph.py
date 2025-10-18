
from dataclasses import dataclass
import torch
from adp_ae_core import AdaptiveDNNNodeAE, TrainCfg, train_early_stop
from dnn_ae_graph import load_planetoid

@dataclass
class WidthOnlyCfg:
    ex_k:int=8; max_width:int=4096; max_neurons:int=200_000_000; delta:float=1e-6; trials_width:int=200

def fit_width_only(dataset="Cora", init_hidden=64, init_depth=2, seed=42):
    torch.manual_seed(seed)
    data,_=load_planetoid(dataset); X,_,_,_,_=data; N,F=X.size(0), X.size(1)
    m=AdaptiveDNNNodeAE(N, init_hidden, init_depth); tcfg=TrainCfg(); cfg=WidthOnlyCfg()
    best,_,_=train_early_stop(m, data, tcfg); best_state=m.snapshot(); baseline=best
    pat=cfg.trials_width
    while pat>0:
        proj=(m.hidden+cfg.ex_k)*(m.depth+1)+m.num_nodes*F
        if proj>cfg.max_neurons or (m.hidden+cfg.ex_k)>cfg.max_width: break
        m.widen_all(cfg.ex_k, F)
        v,_,_=train_early_stop(m, data, tcfg)
        if v<baseline-cfg.delta:
            baseline=v; best_state=m.snapshot()
        else:
            m.restore(best_state); pat-=1
    m.restore(best_state)
    final_val, final_test,_=train_early_stop(m, data, tcfg)
    return m, final_val, final_test

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Cora", choices=["Cora","Citeseer","PubMed"])
    p.add_argument("--init-hidden", type=int, default=64); p.add_argument("--init-depth", type=int, default=2)
    a=p.parse_args()
    model, v, t = fit_width_only(a.dataset, a.init_hidden, a.init_depth)
    print(f"[AE ADP Width-Only] {a.dataset} ValMSE={v:.6f} TestMSE={t:.6f} H={model.hidden} D={model.depth}")

if __name__=="__main__": main()
