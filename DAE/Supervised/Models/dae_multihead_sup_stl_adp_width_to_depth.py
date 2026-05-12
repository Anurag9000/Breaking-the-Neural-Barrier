"""
ADP wrapper for multi-head DAE.

Reuses the group-sparse MLP DAE supervised ADP implementation.
"""

from .dae_groupsparse_mlp_sup_stl_adp_width_to_depth import (  # noqa: F401
    ADPConfig,
    adp_search,
)
from .dae_groupsparse_mlp_sup_stl import SupDAEGroupSparseMLP as ModelClass  # type: ignore # noqa: F401
