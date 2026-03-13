from __future__ import annotations

import numpy as np


def percentile_scores(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=float)
    return (ranks / (len(values) - 1)) * 100.0
