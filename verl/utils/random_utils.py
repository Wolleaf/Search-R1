import random

import numpy as np
import torch


def derive_rank_seed(base_seed: int, rank: int) -> int:
    """Return a stable, distinct 32-bit seed for one distributed rank."""
    base_seed = int(base_seed)
    rank = int(rank)
    if base_seed < 0 or rank < 0:
        raise ValueError("base_seed and rank must be non-negative")
    return (base_seed + rank) % (2**32)


def seed_everything(seed: int) -> None:
    """Seed process-local RNGs once before data loading or model work."""
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
