from DAE.Self-Supervised.Models.dae_tcn_seq_stl_adp_width_to_depth import (  # type: ignore
    ADPConfig,
    adp_search,
    make_loaders,
)
from DAE.Self-Supervised.Models.dae_tcn_seq_stl import DAETCNSeq  # type: ignore

# For sequence tagging with a TCN DAE auxiliary loss, we reuse the ADP search
# for the temporal DAE backbone. Width/depth expansion behaviour is identical
# to the self-supervised temporal DAE, so the supervised variant simply
# aliases the model class and ADP utilities.

ModelClass = DAETCNSeq

