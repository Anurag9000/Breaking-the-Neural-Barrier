from pathlib import Path
import importlib.util
import torch.nn as nn

# Reuse the width_to_depth implementation; just change default mode
BASE_PATH = Path(__file__).with_name("dnn_ae_graph_adp_width_to_depth.py").resolve()
_spec = importlib.util.spec_from_file_location("adp_impl", BASE_PATH)
adp_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adp_impl)


class ADP_DnnAeGraph(adp_impl.ADP_DnnAeGraph):  # type: ignore
    def __init__(self, base_model: nn.Module, adp_mode: str = "depth_to_width", **kwargs):
        super().__init__(base_model, adp_mode=adp_mode, **kwargs)
