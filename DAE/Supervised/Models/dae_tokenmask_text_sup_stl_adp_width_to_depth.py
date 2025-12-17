from DAE.Self-Supervised.Models.dae_tokenmask_text_stl_adp_width_to_depth import (  # type: ignore
    ADPConfig,
    adp_search,
    make_loaders,
)
from DAE.Self-Supervised.Models.dae_tokenmask_text_stl import TokenMaskTransformerDAE  # type: ignore

# Supervised text DAE classifier (BERT-style) reuses the token-masking DAE ADP
# implementation. The supervised head is built on top of the same backbone,
# so width/depth expansion remains identical to the unsupervised setting.

ModelClass = TokenMaskTransformerDAE

