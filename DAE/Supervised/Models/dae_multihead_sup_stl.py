"""
Multi-head DAE: shared encoder with reconstruction + multiple supervised heads.

Implemented as an alias to the group-sparse MLP DAE encoder + classifier;
additional heads can be added in downstream code as needed.
"""

from .dae_groupsparse_mlp_sup_stl import SupDAEGroupSparseMLP as SupDAEMultiHead  # type: ignore # noqa: F401
from .dae_groupsparse_mlp_sup_stl import sup_dae_total_neurons as sup_dae_multihead_total_neurons  # noqa: F401

