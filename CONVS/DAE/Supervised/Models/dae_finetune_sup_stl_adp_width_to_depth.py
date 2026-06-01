"""
ADP wrapper for DAE-pretrained encoder fine-tuning baseline.

Reuses the ADPResNet ADP runner for width/depth expansion.
"""

from CONVS.CNN.ADP_ResNet.run_adp_resnet import ADPConfig, adp_search  # type: ignore # noqa: F401
from CONVS.CNN.ADP_ResNet.adp_resnet_backbone import ADPResNet as ModelClass  # noqa: F401
