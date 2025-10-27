# =============================
# File: run_adp_transformer_speech.py  (RUNNER)
# CLI for speech self-supervision (Wav2Vec2 / HuBERT / WavLM) with 6 ADP algorithms
# =============================

import argparse
import torch
from adp_transformer_speech import SpeechCfg, AdaptiveSpeechSSL, SearchCfg, adp_search

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="wav2vec2", choices=["wav2vec2","hubert","wavlm"])
    p.add_argument("--algo", default="wd", choices=["wd","dw","alt_d","alt_w","depth_only","width_only"])

    # model sizes
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--feat-dim", type=int, default=128)
    p.add_argument("--codebook", type=int, default=100)

    # masking
    p.add_argument("--mask-prob", type=float, default=0.065)
    p.add_argument("--mask-len", type=int, default=10)

    # ADP params
    p.add_argument("--ex-k", type=int, default=32)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)

    # demo training
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--wav-len", type=int, default=16000*2)  # 2s clips
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    cfg = SpeechCfg(
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        feat_dim=args.feat_dim,
        variant=args.variant,
        codebook_size=args.codebook,
        mask_prob=args.mask_prob,
        mask_length=args.mask_len,
    )

    model = AdaptiveSpeechSSL(cfg).to(args.device)

    # synthetic wave batch
    wav = torch.randn(args.batch, args.wav_len, device=args.device)

    s = SearchCfg(
        algo=args.algo,
        ex_k=args.ex_k,
        delta=args.delta,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
    )

    out = adp_search(model, wav, s, lr=args.lr)

    print({
        "best_demo_loss": out["best"],
        "final_d_model": model.cfg.d_model,
        "final_layers": model.cfg.n_layers,
        "neurons": model.total_neurons,
        "variant": args.variant,
        "algo": args.algo,
    })

# Examples
# Wav2Vec2 + wd:   python run_adp_transformer_speech.py --variant wav2vec2 --algo wd
# HuBERT + alt_d:  python run_adp_transformer_speech.py --variant hubert --algo alt_d
# WavLM + dw:      python run_adp_transformer_speech.py --variant wavlm --algo dw
