import random
import numpy as np
import torch


def set_seeds(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
