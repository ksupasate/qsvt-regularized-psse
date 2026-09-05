from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return np.random.default_rng(seed)
