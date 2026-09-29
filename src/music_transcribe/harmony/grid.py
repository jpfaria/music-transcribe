from __future__ import annotations
import numpy as np


def fit_grid(root_changes: list[float], bar_len: float) -> float:
    """Return bar0 in [0, bar_len) such that change times sit closest to k*bar_len + bar0."""
    if not root_changes:
        return 0.0
    ch = np.asarray(root_changes)
    # circular mean of phases
    ph = (ch % bar_len) / bar_len * 2 * np.pi
    mean = np.arctan2(np.sin(ph).mean(), np.cos(ph).mean())
    bar0 = (mean / (2 * np.pi)) * bar_len
    bar0 %= bar_len
    # shift down so bar 1 starts at or before the first change
    while bar0 > ch[0] + 1e-9:
        bar0 -= bar_len
    return float(round(bar0, 3))
