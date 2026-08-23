"""
Reproducibility Utilities
=========================
Helpers for fixing random seeds to ensure strict reproducibility
of all experiments reported in the paper.
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Fix all random seeds for strict reproducibility.

    Sets the seed for Python built-in RNG, NumPy, and PyTorch (CPU and
    all GPUs), and enables deterministic cuDNN behavior.

    Parameters
    ----------
    seed : int, optional
        The random seed to use (default: 42).
    """
    # Python built-in
    random.seed(seed)
    # Hash seed for deterministic dict/set ordering
    os.environ["PYTHONHASHSEED"] = str(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU and GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False