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


def refine_bar_len(root_changes: list[float], bar_len0: float, tol: float = 0.25, min_points: int = 4) -> float:
    """Refine a tempogram bar length by a least-squares fit of bass root-change times on integer bar indices.

    The first change gets index 0. Each later change is measured from the last kept change: ratio
    r = Δt / bar_len0, k = round(r). It is kept with index n_last + k when k ≥ 1 and |r − k| ≤ tol;
    otherwise it is skipped (sub-bar or off-grid change) and the next one is measured from the same
    kept point. With at least min_points kept, bar_len = slope of polyfit(n, t, 1), accepted only when
    0.75 < bar_len / bar_len0 < 1.33; otherwise bar_len0 is returned unchanged.
    """
    if len(root_changes) < min_points or bar_len0 <= 0:
        return bar_len0
    t = [float(x) for x in root_changes]
    ns, ts = [0], [t[0]]
    for ti in t[1:]:
        r = (ti - ts[-1]) / bar_len0
        k = round(r)
        if k < 1 or abs(r - k) > tol:
            continue
        ns.append(ns[-1] + k)
        ts.append(ti)
    if len(ns) < min_points:
        return bar_len0
    bar_len = float(np.polyfit(np.asarray(ns, dtype=float), np.asarray(ts), 1)[0])
    return bar_len if 0.75 < bar_len / bar_len0 < 1.33 else bar_len0
