# Plain Fully Connected MLPs

This repo has a small set of true plain fully connected MLP models outside `Examples/` and the `DNN FOR ...` folders.

## Core Baseline

- [`DAE/DNN/mlp.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/DNN/mlp.py)  
  The simplest plain MLP in the repo. It is a feed-forward stack of `Linear -> BatchNorm1d -> ReLU` blocks with a linear head.

## DAE Family

### Supervised
- [`DAE/Supervised/Models/dae_contractive_mlp_sup_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Supervised/Models/dae_contractive_mlp_sup_stl.py)
- [`DAE/Supervised/Models/dae_groupsparse_mlp_sup_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Supervised/Models/dae_groupsparse_mlp_sup_stl.py)
- [`DAE/Supervised/Models/dae_lowrank_mlp_sup_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Supervised/Models/dae_lowrank_mlp_sup_stl.py)
- [`DAE/Supervised/Models/dae_robust_mlp_sup_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Supervised/Models/dae_robust_mlp_sup_stl.py)
- [`DAE/Supervised/Models/dae_sparse_mlp_sup_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Supervised/Models/dae_sparse_mlp_sup_stl.py)

### Self-Supervised
- [`DAE/Self-Supervised/Models/dae_blockmask_mlp_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Self-Supervised/Models/dae_blockmask_mlp_stl.py)
- [`DAE/Self-Supervised/Models/dae_contractive_mlp_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Self-Supervised/Models/dae_contractive_mlp_stl.py)
- [`DAE/Self-Supervised/Models/dae_groupsparse_mlp_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Self-Supervised/Models/dae_groupsparse_mlp_stl.py)
- [`DAE/Self-Supervised/Models/dae_lowrank_mlp_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Self-Supervised/Models/dae_lowrank_mlp_stl.py)
- [`DAE/Self-Supervised/Models/dae_robust_mlp_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Self-Supervised/Models/dae_robust_mlp_stl.py)
- [`DAE/Self-Supervised/Models/dae_saltpepper_mlp_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Self-Supervised/Models/dae_saltpepper_mlp_stl.py)
- [`DAE/Self-Supervised/Models/dae_sparse_mlp_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Self-Supervised/Models/dae_sparse_mlp_stl.py)
- [`DAE/Self-Supervised/Models/dae_stacked_mlp_stl.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/Self-Supervised/Models/dae_stacked_mlp_stl.py)

## What Is Simplest?

The simplest model in this set is [`DAE/DNN/mlp.py`](/home/anurag-basistha/Projects/Breaking-the-Neural-Barrier/DAE/DNN/mlp.py). It is the cleanest plain feed-forward MLP and is closest to the `Examples/` den-style baselines.
