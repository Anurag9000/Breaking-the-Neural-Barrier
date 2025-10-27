# =============================
# File: run_adp_transformer_contrastive.py (RUNNER)
# CLI for contrastive SSL family (SimCLR, CPC, QuickThought, SimCSE)
# =============================

import argparse
import torch
from adp_transformer_contrastive import ContrastiveCfg, AdaptiveTransformerContrastive, SearchCfg, adp_search

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="simclr", choices=["simclr","cpc","quickthought","simcse"])
    p.add_argument("--algo", default="wd", choices=["wd","dw","alt_d","alt_w","depth_only","width_only"])

    # model params
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--proj-dim", type=int, default=128)

    # ADP params
    p.add_argument("--ex-k", type=int, default=64)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)

    # train demo
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    cfg = ContrastiveCfg(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        projection_dim=args.proj_dim,
        variant=args.variant,
    )

    model = AdaptiveTransformerContrastive(cfg).to(args.device)

    # dummy positive pairs (augmentations) or sequences
    tokens_a = torch.randint(5, args.vocab_size, (args.batch, args.seq_len), device=args.device)
    tokens_b = torch.randint(5, args.vocab_size, (args.batch, args.seq-len), device=args.device)

    s = SearchCfg(
        algo=args.algo,
        ex_k=args.ex_k,
        delta=args.delta,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
    )

    out = adp_search(model, tokens_a, tokens_b, s, lr=args.lr)

    print({
        "best_demo_loss": out["best"],
        "final_d_model": model.cfg.d_model,
        "final_layers": model.cfg.n_layers,
        "neurons": model.total_neurons,
        "variant": args.variant,
        "algo": args.algo,
    })

# Examples
# SimCLR + width→depth:       python run_adp_transformer_contrastive.py --variant simclr --algo wd
# QuickThought + alt-depth:   python run_adp_transformer_contrastive.py --variant quickthought --algo alt_d
# CPC + depth-only:           python run_adp_transformer_contrastive.py --variant cpc --algo depth_only
# SimCSE + width-only:        python run_adp_transformer_contrastive.py --variant simcse --algo width_only
