"""Utilities for deterministic and reproducible experiment runs."""

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Set Python, NumPy, and PyTorch seeds when PyTorch is available.

    The retrieval pipeline does not train models, but neural encoding and
    re-ranking use PyTorch-backed models. Setting these seeds makes the
    environment explicit and improves reproducibility across runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"[INFO] Global random seed set to {seed}")
