from __future__ import annotations
import librosa
import numpy as np
from scipy.signal import detrend
from music_transcribe.schema import Note

BEND_MIN_CENTS = 40.0
VIB_MIN_CENTS = 20.0
VIB_HZ = (4.0, 8.0)


def classify_contour(cents: np.ndarray, hop_s: float) -> str:
    c = np.asarray(cents, dtype=float)
    ok = ~np.isnan(c)
    if ok.sum() < max(8, 0.4 * len(c)):
        return ""
    c = c[ok]
    n = len(c)
    third = max(3, n // 3)
    head, tail = np.median(c[:third]), np.median(c[-third:])
    peak = np.max(c)
    # bend: sustained rise
    if tail - head >= BEND_MIN_CENTS:
        semis = max(1, int(round((tail - head) / 100)))
        return f"bend:{min(semis, 2)}"
    # bend-release: went up and came back
    if peak - head >= BEND_MIN_CENTS * 2 and abs(tail - head) < BEND_MIN_CENTS / 2:
        return "bend-release"
    # vibrato: periodic oscillation around a trend
    detr = detrend(c)
    if np.std(detr) >= VIB_MIN_CENTS / 2:
        crossings = np.sum(np.diff(np.sign(detr)) != 0)
        dur = n * hop_s
        hz = crossings / (2 * dur) if dur > 0 else 0
        amp = (np.percentile(c, 90) - np.percentile(c, 10)) / 2
        if VIB_HZ[0] <= hz <= VIB_HZ[1] and amp >= VIB_MIN_CENTS:
            return "vibrato"
    return ""


def detect_legato(prev: Note, cur: Note, gap_voiced_frac: float, onset_ratio: float) -> str:
    gap = cur.start - prev.end
    if gap > 0.15:
        return ""
    if gap_voiced_frac >= 0.6 and prev.pitch != cur.pitch:
        return f"slide:{cur.pitch}"
    if onset_ratio < 0.5 and abs(cur.pitch - prev.pitch) <= 4:
        return "hammer" if cur.pitch > prev.pitch else "pull"
    return ""


def onset_ratio(y: np.ndarray, sr: int, t: float, window: float = 0.05) -> float:
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=512)
    m = (times >= t - window) & (times <= t + window)
    med = np.median(oenv) + 1e-9
    return float(oenv[m].max() / med) if m.any() else 1.0
