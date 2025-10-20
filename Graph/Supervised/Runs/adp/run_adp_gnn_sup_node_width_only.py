
import argparse, torch
from adp_gnn_sup_node_core import (AdaptiveGNNNode, TrainConfig, SearchConfig,
                                   load_planetoid,
                                   gnn_node_width_to_depth, gnn_node_depth_to_width,
                                   gnn_node_alt_depth_first, gnn_node_alt_width_first,
                                   gnn_node_depth_only, gnn_node_width_only)

def build(parser):
    parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora","CiteSeer","PubMed"])
    parser.add_argument("--data-root", type=str, default="./data")

    parser.add_argument("--conv-type", type=str, default="sage", choices=["sage", "gcn", "gat"])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--init-width", type=int, default=64)
    parser.add_argument("--init-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--es-patience", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=None)

    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--patience-width", type=int, default=2)
    parser.add_argument("--patience-depth", type=int, default=2)
    parser.add_argument("--ex-k", type=int, default=16)
    parser.add_argument("--max-neurons", type=int, default=1200000)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-width", type=int, default=4096)
    parser.add_argument("--max-total-epochs", type=int, default=None)
    return parser

def get_common(args):
    data, in_dim, num_classes = load_planetoid(args.dataset, root=args.data_root)
    widths = [args.init_width] * args.init_depth
    net = AdaptiveGNNNode(in_dim=in_dim, hidden_dims=widths, num_classes=num_classes,
                          conv_type=args.conv_type, heads=args.heads, dropout=args.dropout)
    tcfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, es_patience=args.es_patience,
                       grad_clip=args.grad_clip)
    scfg = SearchConfig(delta=args.delta, patience_width=args.patience_width, patience_depth=args.patience_depth,
                        ex_k=args.ex_k, max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width,
                        max_total_epochs=args.max_total_epochs)
    return net, tcfg, scfg, data

def final_eval(net, data, device):
    import torch
    import torch.nn.functional as F
    net.eval(); net.to(device)
    data = data.to(device)
    with torch.no_grad():
        logits = net(data)
        pred = logits.argmax(dim=1)
        def acc(mask):
            m = mask.bool()
            return float((pred[m] == data.y[m]).sum().item()) / max(int(m.sum().item()), 1)
    print(f"[TEST] acc_test={acc(data.test_mask):.4f} acc_val={acc(data.val_mask):.4f} depth={len(net.hidden_dims)} widths={net.widths} neurons={net.total_neurons()}")

def main():
    parser = build(argparse.ArgumentParser())
    args = parser.parse_args()
    net, tcfg, scfg, data = get_common(args)
    net = gnn_node_width_only(net, data, tcfg, scfg, max_epochs=args.max_epochs)
    final_eval(net, data, tcfg.device)
if __name__ == "__main__": main()
