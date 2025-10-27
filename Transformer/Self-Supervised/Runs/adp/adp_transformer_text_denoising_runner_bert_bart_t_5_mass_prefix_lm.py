# =============================
# File: run_adp_transformer_text.py  (RUNNER)
# CLI to pick variant + ADP algorithm for text denoising/masked-token tasks
# =============================

import argparse
import torch
from adp_transformer_text import ADPTextCfg, AdaptiveTransformerText, SearchCfg, adp_search

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="mlm_bert", choices=["mlm_bert","bart","t5_span","mass","prefixlm"])
    p.add_argument("--algo", default="wd", choices=["wd","dw","alt_d","alt_w","depth_only","width_only"])

    # model sizes
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers-enc", type=int, default=6)
    p.add_argument("--layers-dec", type=int, default=6)

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

    cfg = ADPTextCfg(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers_enc=args.layers_enc,
        n_layers_dec=args.layers_dec,
        variant=args.variant,
    )

    model = AdaptiveTransformerText(cfg).to(args.device)

    # --- demo synthetic tokens (replace with real dataset/dataloader)
    tokens = torch.randint(5, args.vocab_size, (args.batch, args.seq_len), device=args.device)

    s = SearchCfg(
        algo=args.algo,
        ex_k=args.ex_k,
        delta=args.delta,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
    )

    out = adp_search(model, tokens, s, lr=args.lr)
    print({
        "best_demo_loss": out["best"],
        "final_d_model": model.cfg.d_model,
        "final_enc_layers": model.cfg.n_layers_enc,
        "final_dec_layers": model.cfg.n_layers_dec,
        "neurons": model.total_neurons,
        "variant": args.variant,
        "algo": args.algo,
    })

# Example runs
# BERT MLM + width→depth:   python run_adp_transformer_text.py --variant mlm_bert --algo wd
# BART denoise + alt depth: python run_adp_transformer_text.py --variant bart --algo alt_d
# T5 span-infilling + dw:   python run_adp_transformer_text.py --variant t5_span --algo dw
# MASS seq2seq + depth-only:python run_adp_transformer_text.py --variant mass --algo depth_only
# Prefix-LM + width-only:   python run_adp_transformer_text.py --variant prefixlm --algo width_only
