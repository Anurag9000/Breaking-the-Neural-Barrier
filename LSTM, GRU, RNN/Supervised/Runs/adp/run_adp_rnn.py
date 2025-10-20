
import argparse
import torch
from torch.utils.data import DataLoader, random_split

# Local imports
from adp_rnn import (
    TextRNN, ADP_RNN_Search, TrainCfg, SearchCfg, TensorTextDataset, evaluate
)

# ---------------------------
# Toy tokenizer + dataset
# ---------------------------

def build_toy_dataset(num_samples=4000, vocab_size=5000, seq_len=64, num_classes=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(1, vocab_size, (num_samples, seq_len), generator=g)
    y = torch.randint(0, num_classes, (num_samples,), generator=g)
    lengths = torch.full((num_samples,), seq_len, dtype=torch.long)
    return TensorTextDataset(X, y, lengths), vocab_size, num_classes

def split_dataset(ds, val_frac=0.1, test_frac=0.1, seed=0):
    n = len(ds)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test
    return random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))

# ---------------------------
# Main
# ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="width_to_depth",
                   choices=["width_to_depth","depth_to_width","alt_width","alt_depth","width_only","depth_only"])
    p.add_argument("--vocab_size", type=int, default=5000)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--toy_samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)

    # Model/Train/Search knobs
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--nonlinearity", type=str, default="tanh", choices=["tanh","relu"])

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--es_patience", type=int, default=10)

    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--ex_k", type=int, default=32)
    p.add_argument("--trials_width", type=int, default=50)
    p.add_argument("--trials_depth", type=int, default=50)
    p.add_argument("--max_total_epochs", type=int, default=300)
    p.add_argument("--max_layers", type=int, default=12)
    p.add_argument("--max_hidden", type=int, default=2048)

    args = p.parse_args()

    # Dataset (toy synthetic for plug-and-play; replace with real tokenizer/loader as needed)
    ds, vocab_size, num_classes = build_toy_dataset(
        num_samples=args.toy_samples,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        seed=args.seed
    )
    train_ds, val_ds, test_ds = split_dataset(ds, 0.1, 0.1, args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    # Model
    model = TextRNN(
        vocab_size=vocab_size,
        num_classes=num_classes,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden,
        num_layers=args.layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        nonlinearity=args.nonlinearity,
        pad_idx=0,
    )

    train_cfg = TrainCfg(lr=args.lr, weight_decay=args.wd, batch_size=args.batch, max_epochs=args.max_epochs, es_patience=args.es_patience)
    search_cfg = SearchCfg(delta=args.delta, ex_k=args.ex_k, trials_width=args.trials_width, trials_depth=args.trials_depth,
                           max_total_epochs=args.max_total_epochs, max_layers=args.max_layers, max_hidden=args.max_hidden)

    adp = ADP_RNN_Search(model, train_cfg, search_cfg)

    # Pick strategy
    if args.mode == "width_to_depth":
        adp.search_width_to_depth(train_loader, val_loader)
    elif args.mode == "depth_to_width":
        adp.search_depth_to_width(train_loader, val_loader)
    elif args.mode == "alt_width":
        adp.search_alt_width_first(train_loader, val_loader)
    elif args.mode == "alt_depth":
        adp.search_alt_depth_first(train_loader, val_loader)
    elif args.mode == "width_only":
        adp.search_width_only(train_loader, val_loader)
    elif args.mode == "depth_only":
        adp.search_depth_only(train_loader, val_loader)
    else:
        raise ValueError("Unknown mode")

    # Final evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = adp.model.to(device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[RESULT] Test loss: {test_loss:.4f}  |  Test acc: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
