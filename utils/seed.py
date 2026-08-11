"""Reproducibility utilities"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed
        deterministic: Use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: Some operations don't have deterministic implementations
        torch.use_deterministic_algorithms(True, warn_only=True)
