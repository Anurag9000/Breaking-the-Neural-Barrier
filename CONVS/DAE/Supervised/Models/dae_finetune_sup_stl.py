"""
DAE-pretrained encoder followed by pure supervised fine-tuning baseline.

We alias this to the plain ADPResNet STL classifier; DAE pretraining is
handled by running an appropriate DAE STL/ADP first and then fine-tuning.
"""

from CONVS.CNN.ADP_ResNet.adp_resnet_backbone import ADPResNet as SupDAEFinetune  # noqa: F401
from CONVS.CNN.ADP_ResNet.adp_resnet_backbone import estimate_neurons as sup_dae_finetune_total_neurons  # noqa: F401

