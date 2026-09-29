from __future__ import annotations
import librosa
import numpy as np


def _tempo_scores(oenv: np.ndarray, sr: int) -> list[tuple[float, float]]:
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
    ac = tg.mean(axis=1)
    bpms = librosa.tempo_frequencies(len(ac), sr=sr)
    idx = [i for i in np.argsort(ac)[::-1] if 30 < bpms[i] < 240][:8]
    return [(float(bpms[i]), float(ac[i])) for i in idx]


def _tempo_candidates(oenv: np.ndarray, sr: int) -> list[float]:
    return [b for b, _ in _tempo_scores(oenv, sr)]


def _octave_correct_low_evidence(scored: list[tuple[float, float]], lo: float = 40.0, hi: float = 100.0) -> float:
    """Pick a plausible tempo when there are too few chord changes to size the
    bar directly. A tempogram peak and its octave/triplet-ratio siblings
    (x1.5, x2, x3, x4 of every candidate) are pooled and weighted by the
    candidate's own tempogram strength; the family of mutually-consistent
    readings with the most corroborating strength in the plausible tempo
    range wins the vote. A fixed "closest to some single center" rule can't
    tell a real fundamental from its 1.5x sibling when both land near the
    center, but the fundamental's family is backed by more evidence.
    """
    factors = (1.0, 1.5, 2.0, 3.0, 4.0)
    pool: list[tuple[float, float]] = []
    for b, w in scored:
        for f in factors:
            pool.append((round(b / f, 2), w))
        pool.append((round(b * 2, 2), w))
    in_range = sorted(p for p in pool if lo <= p[0] <= hi)
    if not in_range:
        return scored[0][0]
    tol = 3.0
    clusters = [[in_range[0]]]
    for p in in_range[1:]:
        if p[0] - clusters[-1][-1][0] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    best = max(clusters, key=lambda c: sum(w for _, w in c))
    total_w = sum(w for _, w in best)
    return sum(v * w for v, w in best) / total_w


def _meter(oenv: np.ndarray, sr: int, bpm: float) -> str:
    hop_s = 512 / sr
    beat = 60.0 / bpm
    ac = librosa.autocorrelate(oenv)
    def at(lag_s: float) -> float:
        k = int(round(lag_s / hop_s))
        return float(ac[k]) if 0 < k < len(ac) else 0.0
    triplet = at(beat / 3) + at(2 * beat / 3)
    duple = at(beat / 2) + at(beat / 4)
    return "12/8" if triplet > duple else "4/4"


def estimate_tempo(y: np.ndarray, sr: int, root_changes: list[float]) -> tuple[float, str]:
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    cands = _tempo_candidates(oenv, sr)
    if not cands:
        # nothing to analyze: fall back to the most common default rather than
        # crashing on an empty tempogram
        return 120.0, "4/4"
    if len(root_changes) >= 3:
        # expand each candidate by integer factors so octave errors are recoverable
        expanded = sorted({round(c / f, 2) for c in cands for f in (1, 2, 3, 4)} | {round(c * 2, 2) for c in cands})
        expanded = [b for b in expanded if 30 <= b <= 240]
        target_bar = float(np.median(np.diff(root_changes)))
        def err(b: float) -> float:
            bar = 4 * 60.0 / b
            # allow one chord per bar or per two bars
            return min(abs(bar - target_bar), abs(2 * bar - target_bar)) / target_bar
        bpm = min(expanded, key=err)
    else:
        # not enough chord changes to size the bar directly from bass root
        # timing: fall back to a strength-weighted octave vote instead of
        # bypassing octave correction entirely
        bpm = _octave_correct_low_evidence(_tempo_scores(oenv, sr))
    return round(bpm, 1), _meter(oenv, sr, bpm)
