
import argparse, torch
from adp_gnn_sup_core import (AdaptiveGNN_Sup, TrainConfig, SearchConfig,
                              make_tu_loader,
                              gnn_sup_width_to_depth, gnn_sup_depth_to_width,
                              gnn_sup_alt_depth_first, gnn_sup_alt_width_first,
                              gnn_sup_depth_only, gnn_sup_width_only)

def build(parser):
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)

    parser.add_argument("--conv-type", type=str, default="sage", choices=["sage", "gcn", "gat"])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--init-width", type=int, default=64)
    parser.add_argument("--init-depth", type=int, default=3)
    parser.add_argument("--pool-idx", type=int, nargs="*", default=[])
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--es-patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--patience-width", type=int, default=2)
    parser.add_argument("--patience-depth", type=int, default=2)
    parser.add_argument("--ex-k", type=int, default=16)
    parser.add_argument("--max-neurons", type=int, default=1200000)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=2048)
    parser.add_argument("--max-total-epochs", type=int, default=None)
    return parser

def get_common(args):
    dl_train, dl_val, dl_test, in_dim, num_classes = make_tu_loader(args.dataset, args.batch_size, args.num_workers, args.val_split, root=args.data_root)
    widths = [args.init_width] * args.init_depth
    net = AdaptiveGNN_Sup(in_dim=in_dim, hidden_dims=widths, num_classes=num_classes,
                          conv_type=args.conv_type, heads=args.heads,
                          pooling_indices=args.pool_idx, dropout=args.dropout)
    tcfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, es_patience=args.es_patience,
                       grad_clip=args.grad_clip)
    scfg = SearchConfig(delta=args.delta, patience_width=args.patience_width, patience_depth=args.patience_depth,
                        ex_k=args.ex_k, max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width,
                        max_total_epochs=args.max_total_epochs, pooling_indices=tuple(args.pool_idx))
    return net, tcfg, scfg, dl_train, dl_val, dl_test

def final_eval(net, dl_test, device):
    import torch, torch.nn.functional as F
    net.eval(); net.to(device)
    tot, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for batch in dl_test:
            batch = batch.to(device)
            logits = net(batch)
            y = batch.y.view(-1).to(device)
            loss = F.cross_entropy(logits, y, reduction="sum")
            tot += float(loss.item()); n += batch.num_graphs
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
    print(f"[TEST] loss={tot/max(n,1):.6f}  acc={correct/max(n,1):.4f}  neurons={net.total_neurons()}  depth={len(net.hidden_dims)}  widths={net.widths}")

def main():
    parser = build(argparse.ArgumentParser())
    args = parser.parse_args()
    net, tcfg, scfg, dl_train, dl_val, dl_test = get_common(args)
    net = gnn_sup_depth_to_width(net, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)
    final_eval(net, dl_test, tcfg.device)
if __name__ == "__main__": main()
