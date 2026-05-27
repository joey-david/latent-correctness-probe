from __future__ import annotations

import random

# Central experiment configuration shared across modules.
CHECKPOINT_STEPS = [4, 8, 16, 32, 64, 128, 192, 256, 384, 512]
MAX_PREFIX_TOKENS = 1024  # allow analysing longer reasoning prefixes when needed
MAX_NEW_TOKENS = 512  # limit chain-of-thought length before forcing a conclusion
DEVICE = "cuda"
SEED = 356


def set_global_seed(seed: int = SEED) -> None:
    """Ensure reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


# Initialise module-level state upon import.
set_global_seed(SEED)
