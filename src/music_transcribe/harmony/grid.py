from __future__ import annotations
import numpy as np


def fit_grid(root_changes: list[float], bar_len: float) -> float:
    """Return bar0 such that change times sit closest to k*bar_len + bar0.

    bar0 is in (root_changes[0] - 0.75*bar_len, root_changes[0] + 0.25*bar_len];
    it may be negative when the first bar is a partial pickup (song starts mid-bar).
    A pickup shorter than 75% of a bar is plausible; a "pickup" of 75% or more of a
    bar is treated as a phase-wrap artifact and folded back by one bar_len instead.
    """
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
    # a near-full-bar pickup is almost certainly a phase-wrap artifact, not a real pickup
    if ch[0] - bar0 >= 0.75 * bar_len:
        bar0 += bar_len
    return float(round(bar0, 3))
