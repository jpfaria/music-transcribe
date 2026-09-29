from __future__ import annotations
import numpy as np


def fit_grid(root_changes: list[float], bar_len: float) -> float:
    """Return bar0 such that change times sit closest to k*bar_len + bar0.

    bar0 ∈ (−bar_len, 0.05·bar_len]: bar 1 always covers t=0, so no audio before
    the first detected change (e.g. an intro before the bass enters) is dropped.
    A negative bar0 means bar 1 is a partial (pickup) bar; a phase within 5% of a
    bar after 0 is treated as the song starting on the downbeat.
    """
    if not root_changes:
        return 0.0
    ch = np.asarray(root_changes)
    # circular mean of phases
    ph = (ch % bar_len) / bar_len * 2 * np.pi
    mean = np.arctan2(np.sin(ph).mean(), np.cos(ph).mean())
    bar0 = ((mean / (2 * np.pi)) * bar_len) % bar_len
    if bar0 > 0.05 * bar_len:
        bar0 -= bar_len
    return float(round(bar0, 3))
