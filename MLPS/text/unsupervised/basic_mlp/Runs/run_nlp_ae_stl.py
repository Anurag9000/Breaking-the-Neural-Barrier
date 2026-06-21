
import argparse, os, random, torch
from torch.utils.data import DataLoader
from nlp_utils_unsup import TextOnlyCSV, build_vocab_from_csv, collate_unsup
from nlp_ae_stl import MLPTextAE
from nlp_ae_common import soft_ce_loss

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True, help="CSV with header: text")
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--rep_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_size", type=int, default=50000)
    p.add_argument("--word_dropout", type=float, default=0.1)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = build_vocab_from_csv(args.train_csv, min_freq=args.min_freq, max_size=args.max_size)
    tr = TextOnlyCSV(args.train_csv); va = TextOnlyCSV(args.val_csv)
    collate = lambda batch: collate_unsup(batch, vocab, args.max_len, view_word_dropout=args.word_dropout)
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate)
    val = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)

    model = MLPTextAE(len(vocab), args.emb_dim, args.hidden, args.rep_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(1, args.epochs+1):
        model.train(); tr_sum, tr_n = 0.0, 0
        for (tok, lens), bow in trl:
            tok, lens, bow = tok.to(device), lens.to(device), bow.to(device)
            logits = model((tok, lens))
            loss = soft_ce_loss(logits, bow)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item() * tok.size(0); tr_n += tok.size(0)
        tr_loss = tr_sum / max(tr_n, 1)

        model.eval(); va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for (tok, lens), bow in val:
                tok, lens, bow = tok.to(device), lens.to(device), bow.to(device)
                logits = model((tok, lens))
                loss = soft_ce_loss(logits, bow)
                va_sum += loss.item() * tok.size(0); va_n += tok.size(0)
        va_loss = va_sum / max(va_n, 1)

        if va_loss < best_val:
            best_val, best_state, bad = va_loss, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1

        print(f"[{ep:03d}] train_rec={tr_loss:.6f} val_rec={va_loss:.6f} | hidden={args.hidden} rep={args.rep_dim}")
        if bad >= args.patience: break

    if best_state is not None:
        os.makedirs("checkpoints", exist_ok=True)
        path = os.path.join("checkpoints", "nlp_ae_stl.pt")
        torch.save({"model": best_state, "val_rec": best_val, "config": vars(args)}, path)
        print(f"Saved best checkpoint to: {path} (val_rec={best_val:.6f})")

if __name__ == "__main__":
    try:
        import os as _os, sys as _sys
        if _os.name == "posix" and _sys.platform.startswith("linux"):
            import ctypes as _ctypes
            _ctypes.CDLL("libc.so.6", use_errno=True).mlockall(3)
        elif _os.name == "nt":
            import ctypes as _ctypes
            _ctypes.windll.kernel32.SetProcessWorkingSetSize(_ctypes.windll.kernel32.GetCurrentProcess(), -1, -1)
    except Exception:
        pass
    main()
