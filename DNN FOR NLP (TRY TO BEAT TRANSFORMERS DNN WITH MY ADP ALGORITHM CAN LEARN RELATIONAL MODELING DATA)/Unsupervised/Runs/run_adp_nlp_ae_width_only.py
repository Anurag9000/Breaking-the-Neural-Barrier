
import argparse, os, random, torch
from torch.utils.data import DataLoader
from adp_nlp_ae_width_only import AdaptiveTextAE, adp_search_width_only
from nlp_utils_unsup import TextOnlyCSV, build_vocab_from_csv, collate_unsup

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--rep_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_size", type=int, default=50000)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--ex_k", type=int, default=128)
    p.add_argument("--trials_width", type=int, default=3)
    p.add_argument("--max_neurons", type=int, default=32768)
    p.add_argument("--max_depth", type=int, default=16)
    p.add_argument("--max_width", type=int, default=4096)
    p.add_argument("--word_dropout", type=float, default=0.1)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = build_vocab_from_csv(args.train_csv, min_freq=args.min_freq, max_size=args.max_size)
    tr = TextOnlyCSV(args.train_csv); va = TextOnlyCSV(args.val_csv)
    collate = lambda batch: collate_unsup(batch, vocab, args.max_len, view_word_dropout=args.word_dropout)
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate, drop_last=False)
    val = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate, drop_last=False)

    model = AdaptiveTextAE(len(vocab), args.emb_dim, args.hidden, args.rep_dim, use_bn=True)

    best = adp_search_width_only(model, trl, val, device,
                                 trials_width=args.trials_width,
                                 epochs=args.epochs, lr=args.lr, patience=args.patience,
                                 delta=args.delta, ex_k=args.ex_k,
                                 max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width)

    os.makedirs("checkpoints", exist_ok=True)
    path = os.path.join("checkpoints", "adp_nlp_ae_width_only.pt")
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "best_val_rec": best, "config": vars(args)}, path)
    print(f"Saved best model to {path} (val_rec={best:.6f})")

if __name__ == "__main__":
    main()
