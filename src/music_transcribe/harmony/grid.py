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


def refine_bar_len(root_changes: list[float], bar_len0: float, tol: float = 0.25, min_intervals: int = 4) -> float:
    """Refine a tempogram bar length from the bass root-change intervals.

    Each interval iv between consecutive changes is read as k bars, k = round(iv / bar_len0) ∈ {1..4},
    and kept when |iv / bar_len0 − k| ≤ tol. With at least min_intervals kept, returns median(iv / k);
    otherwise bar_len0 unchanged. A few % of tempo error drifts the grid a whole bar within ~25 bars,
    while the harmonic rhythm pins the bar directly.
    """
    if len(root_changes) < 2 or bar_len0 <= 0:
        return bar_len0
    iv = np.diff(np.asarray(root_changes, dtype=float))
    r = iv / bar_len0
    k = np.round(r)
    ok = (k >= 1) & (k <= 4) & (np.abs(r - k) <= tol)
    if ok.sum() < min_intervals:
        return bar_len0
    return float(np.median(iv[ok] / k[ok]))
