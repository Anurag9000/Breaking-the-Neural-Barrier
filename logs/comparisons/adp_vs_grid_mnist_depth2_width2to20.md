# ADP vs Grid (MNIST, depth=2, width=2..20 step 2)

- ADP CSV: `logs/adp_mnist_mlp_classifier/mnist_width_only_d2_w1_exk1_20251219_180230/training_stats.csv`
- Grid summary: `logs/grid_mnist_depth2_width2to20_step2_parallel10_gpu/mnist_grid_depth2_w2-20_step2_20251219_210045/grid_summary.csv`
- Comparison CSV: `logs/comparisons/adp_vs_grid_mnist_depth2_width2to20.csv`

## Best (by validation loss)
- ADP best: width=18 val_loss=0.193559 best_val_acc=0.952833
- Grid best: width=20 val_loss=0.133262 best_val_acc=0.960333

## Best (by validation accuracy)
- ADP best: width=18 best_val_acc=0.952833 val_loss=0.193559
- Grid best: width=20 best_val_acc=0.960333 val_loss=0.133262

## Same architecture (width=18, depth=2)
- ADP: val_loss=0.193559 val_acc@best_loss=0.949833 best_val_acc=0.952833
- Grid: val_loss=0.154226 val_acc@best_loss=0.953333 best_val_acc=0.955000
- Deltas (ADP-Grid): val_loss=0.039332 best_val_acc=-0.002167
