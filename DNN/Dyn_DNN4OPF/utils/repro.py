import torch
import random
import numpy as np

def set_determinism(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)  

def worker_init_fn(worker_id):
    # Derive a unique but deterministic seed per worker
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
