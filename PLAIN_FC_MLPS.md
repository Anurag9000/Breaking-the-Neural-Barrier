# Plain Fully Connected MLPs

This repo centralizes plain fully connected MLP models under `MLPS/`.

## Core Baseline

- [`MLPS/tabular/shared/dae_dnn/mlp.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/tabular/shared/dae_dnn/mlp.py)
  The simplest plain MLP in the repo. It is a feed-forward stack of `Linear -> BatchNorm1d -> ReLU` blocks with a linear head.

## DAE Family

### Supervised
- [`MLPS/image/supervised/dae/Models/dae_contractive_mlp_sup_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/supervised/dae/Models/dae_contractive_mlp_sup_stl.py)
- [`MLPS/image/supervised/dae/Models/dae_groupsparse_mlp_sup_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/supervised/dae/Models/dae_groupsparse_mlp_sup_stl.py)
- [`MLPS/image/supervised/dae/Models/dae_lowrank_mlp_sup_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/supervised/dae/Models/dae_lowrank_mlp_sup_stl.py)
- [`MLPS/image/supervised/dae/Models/dae_robust_mlp_sup_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/supervised/dae/Models/dae_robust_mlp_sup_stl.py)
- [`MLPS/image/supervised/dae/Models/dae_sparse_mlp_sup_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/supervised/dae/Models/dae_sparse_mlp_sup_stl.py)

### Self-Supervised
- [`MLPS/image/unsupervised/dae/Models/dae_blockmask_mlp_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/unsupervised/dae/Models/dae_blockmask_mlp_stl.py)
- [`MLPS/image/unsupervised/dae/Models/dae_contractive_mlp_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/unsupervised/dae/Models/dae_contractive_mlp_stl.py)
- [`MLPS/image/unsupervised/dae/Models/dae_groupsparse_mlp_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/unsupervised/dae/Models/dae_groupsparse_mlp_stl.py)
- [`MLPS/image/unsupervised/dae/Models/dae_lowrank_mlp_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/unsupervised/dae/Models/dae_lowrank_mlp_stl.py)
- [`MLPS/image/unsupervised/dae/Models/dae_robust_mlp_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/unsupervised/dae/Models/dae_robust_mlp_stl.py)
- [`MLPS/image/unsupervised/dae/Models/dae_saltpepper_mlp_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/unsupervised/dae/Models/dae_saltpepper_mlp_stl.py)
- [`MLPS/image/unsupervised/dae/Models/dae_sparse_mlp_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/unsupervised/dae/Models/dae_sparse_mlp_stl.py)
- [`MLPS/image/unsupervised/dae/Models/dae_stacked_mlp_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/image/unsupervised/dae/Models/dae_stacked_mlp_stl.py)

## Graph Family

### Supervised
- [`MLPS/graph/supervised/basic_mlp/Models/dnn_stl_graph.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/graph/supervised/basic_mlp/Models/dnn_stl_graph.py)
- [`MLPS/graph/supervised/basic_mlp/Models/dnn_stl_graph_adp_width_to_depth.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/graph/supervised/basic_mlp/Models/dnn_stl_graph_adp_width_to_depth.py)

### Unsupervised
- [`MLPS/graph/unsupervised/basic_mlp/Models/dnn_ae_graph.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/graph/unsupervised/basic_mlp/Models/dnn_ae_graph.py)
- [`MLPS/graph/unsupervised/basic_mlp/Models/dnn_ae_graph_adp_width_to_depth.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/graph/unsupervised/basic_mlp/Models/dnn_ae_graph_adp_width_to_depth.py)
- [`MLPS/graph/unsupervised/dae/Models/dae_graph_link_stl.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/graph/unsupervised/dae/Models/dae_graph_link_stl.py)
- [`MLPS/graph/unsupervised/dae/Models/dae_graph_link_stl_adp_width_to_depth.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/graph/unsupervised/dae/Models/dae_graph_link_stl_adp_width_to_depth.py)

## What Is Simplest?

The simplest model in this set is [`MLPS/tabular/shared/dae_dnn/mlp.py`](/home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier/MLPS/tabular/shared/dae_dnn/mlp.py). It is the cleanest plain feed-forward MLP and is closest to the `MLPS/examples/` den-style baselines.
