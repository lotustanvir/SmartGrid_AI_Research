"""Global reproducible seeding.

Centralises RNG seeding for ``random``, ``numpy`` and (if available)
``torch`` so every model and experiment entry-point shares one seed path.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic_torch: bool = True) -> int:
    """Seed all supported RNGs and return the seed used.

    Args:
        seed: Non-negative integer seed.
        deterministic_torch: If True and torch is installed, enable
            deterministic algorithms where possible (CPU-safe).

    Raises:
        ValueError: If ``seed`` is not a non-negative int.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("torch not installed; seeded random + numpy only.")

    logger.info("Global seed set to %d", seed)
    return seed
