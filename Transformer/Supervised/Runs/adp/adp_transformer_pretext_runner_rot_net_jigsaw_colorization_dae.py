# =============================
# File: run_adp_transformer_pretext.py  (RUNNER)
# CLI for pretext SSL (RotNet, Jigsaw, Colorization, DAE) with 6 ADP algorithms
# =============================

import argparse
import torch
from adp_transformer_pretext import PretextCfg, AdaptiveViTPretext, SearchCfg, adp_search

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="rotnet", choices=["rotnet","jigsaw","color","dae"])
    p.add_argument("--algo", default="wd", choices=["wd","dw","alt_d","alt_w","depth_only","width_only"])

    # model
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--jigsaw-grid", type=int, default=3)
    p.add_argument("--jigsaw-k", type=int, default=30)

    # adp
    p.add_argument("--ex-k", type=int, default=32)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)

    # train demo
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    cfg = PretextCfg(
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        variant=args.variant,
        jigsaw_grid=args.jigsaw_grid,
        jigsaw_k=args.jigsaw_k,
    )

    model = AdaptiveViTPretext(cfg).to(args.device)

    # synthetic images (replace with real loader)
    imgs = torch.randn(args.batch, 3, args.image_size, args.image_size, device=args.device)

    s = SearchCfg(
        algo=args.algo,
        ex_k=args.ex_k,
        delta=args.delta,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
    )

    out = adp_search(model, imgs, s, lr=args.lr)

    print({
        "best_demo_loss": out["best"],
        "final_dim": model.cfg.dim,
        "final_depth": model.cfg.depth,
        "neurons": model.total_neurons,
        "variant": args.variant,
        "algo": args.algo,
    })

# Examples
# RotNet + wd:      python run_adp_transformer_pretext.py --variant rotnet --algo wd
# Jigsaw + alt_d:   python run_adp_transformer_pretext.py --variant jigsaw --algo alt_d
# Colorization + dw:python run_adp_transformer_pretext.py --variant color --algo dw
# DAE + width-only: python run_adp_transformer_pretext.py --variant dae --algo width_only
