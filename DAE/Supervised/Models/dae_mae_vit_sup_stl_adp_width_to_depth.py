from DAE.Self-Supervised.Models.dae_mae_vit_stl_adp_width_to_depth import (  # type: ignore
    ADPConfig,
    adp_search,
    make_loaders,
)
from DAE.Self-Supervised.Models.dae_mae_vit_stl import MAEViT  # type: ignore

# Supervised MAE-ViT classifier reuses the unsupervised MAE ADP search:
# - same backbone MAEViT.
# - ADP controls embed_dim ("width") and depth exactly as in the self-supervised
#   setting. Here we alias the symbols so DAE/Supervised can invoke identical
#   width/depth expansion logic.

ModelClass = MAEViT

