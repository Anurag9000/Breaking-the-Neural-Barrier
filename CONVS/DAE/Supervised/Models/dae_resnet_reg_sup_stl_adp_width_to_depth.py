from CONVS.CNN.ADP_ResNet.run_adp_resnet import (  # type: ignore
    ADPConfig,
    adp_search,
    make_cifar_transforms,
)
from CONVS.CNN.ADP_ResNet.adp_resnet_backbone import ADPResNet  # type: ignore

# For the DAE-regularised ResNet we reuse the ADP search machinery from
# CNN.ADP_ResNet. The underlying backbone already supports width/depth
# expansion and the runner's loss can be interpreted as classification +
# auxiliary regularisation. Here we simply alias the symbols so the DAE
# supervised namespace can invoke the same ADP logic.

ModelClass = ADPResNet
