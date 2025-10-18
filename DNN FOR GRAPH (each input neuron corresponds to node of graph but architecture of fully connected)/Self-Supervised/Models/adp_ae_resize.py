
import torch, torch.nn as nn

@torch.no_grad()
def _resize_linear(src: nn.Linear, in_features: int, out_features: int) -> nn.Linear:
    dst = nn.Linear(in_features, out_features, bias=(src.bias is not None))
    # overlap copy
    min_out = min(out_features, src.out_features)
    min_in  = min(in_features,  src.in_features)
    dst.weight[:min_out, :min_in].copy_(src.weight[:min_out, :min_in])
    if src.bias is not None and dst.bias is not None:
        dst.bias[:min_out].copy_(src.bias[:min_out])
    # init new rows/cols
    if out_features > min_out:
        nn.init.kaiming_normal_(dst.weight[min_out:out_features, :], nonlinearity="relu")
        if dst.bias is not None: nn.init.zeros_(dst.bias[min_out:out_features])
    if in_features > min_in:
        nn.init.kaiming_normal_(dst.weight[:, min_in:in_features], nonlinearity="relu")
    return dst
