
import argparse, torch
import adp_gnn_ssl_core as core
from adp_gnn_ssl_core import (AdaptiveGNN, TrainConfig, SearchConfig,
                              make_graph_loader,
                              gnn_width_to_depth, gnn_depth_to_width,
                              gnn_alt_depth_first, gnn_alt_width_first,
                              gnn_depth_only, gnn_width_only)

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
    parser.add_argument("--proj-dim", type=int, default=128)

    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--es-patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--lambda-ntx", type=float, default=1.0)
    parser.add_argument("--lambda-barlow", type=float, default=0.0)
    parser.add_argument("--lambda-recon", type=float, default=0.0)
    parser.add_argument("--feat-drop", type=float, default=0.2)
    parser.add_argument("--edge-drop", type=float, default=0.2)
    parser.add_argument("--node-drop", type=float, default=0.0)

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
    dl_train, dl_val, dl_test, in_dim = make_graph_loader(args.dataset, args.batch_size, args.num_workers, args.val_split, root=args.data_root)
    widths = [args.init_width] * args.init_depth
    net = AdaptiveGNN(in_dim=in_dim, hidden_dims=widths, conv_type=args.conv_type, heads=args.heads,
                      pooling_indices=args.pool_idx, dropout=args.dropout,
                      proj_dim=args.proj_dim)
    tcfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, es_patience=args.es_patience,
                       grad_clip=args.grad_clip, temperature=args.temperature,
                       lambda_ntx=args.lambda_ntx, lambda_barlow=args.lambda_barlow, lambda_recon=args.lambda_recon,
                       proj_dim=args.proj_dim, feat_drop=args.feat_drop, edge_drop=args.edge_drop, node_drop=args.node_drop)
    scfg = SearchConfig(delta=args.delta, patience_width=args.patience_width, patience_depth=args.patience_depth,
                        ex_k=args.ex_k, max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width,
                        max_total_epochs=args.max_total_epochs, pooling_indices=tuple(args.pool_idx))
    return net, tcfg, scfg, dl_train, dl_val, dl_test

def final_eval(net, dl_test, device):
    net.eval(); net.to(device)
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in dl_test:
            batch = batch.to(device)
            d1, d2 = core.two_view_augment(batch)
            d1 = d1.to(device); d2 = d2.to(device)
            _, z1 = net(d1); _, z2 = net(d2)
            loss = core.nt_xent(z1, z2)
            tot += float(loss.item()) * batch.num_graphs; n += batch.num_graphs
    print(f"[TEST] ssl_ntx={tot/max(n,1):.6f}  neurons={net.total_neurons()}  depth={len(net.hidden_dims)}  widths={net.widths}")

def main():
    parser = build(argparse.ArgumentParser())
    args = parser.parse_args()
    net, tcfg, scfg, dl_train, dl_val, dl_test = get_common(args)
    net = gnn_depth_only(net, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)
    final_eval(net, dl_test, tcfg.device)
if __name__ == "__main__": main()
