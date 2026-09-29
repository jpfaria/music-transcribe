from __future__ import annotations
from dataclasses import dataclass
import librosa
import numpy as np


@dataclass
class RootRun:
    start: float
    pc: int
    dur: float


def bass_roots(y: np.ndarray, sr: int, step: float = 0.05, min_dur: float = 0.75,
               fmin: float = 30.0, fmax: float = 200.0) -> list[RootRun]:
    hop = 1024
    f0, vf, vp = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=4096, hop_length=hop)
    t = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    midi = np.round(12 * np.log2(np.nan_to_num(f0, nan=1.0) / 440.0) + 69).astype(int)
    pc = np.where(vf, midi % 12, -1)
    n_bins = int(t[-1] / step) + 1 if len(t) else 0
    runs: list[RootRun] = []
    for i in range(n_bins):
        m = (t >= i * step) & (t < (i + 1) * step)
        v = pc[m]; v = v[v >= 0]
        r = int(np.bincount(v, minlength=12).argmax()) if len(v) >= max(1, 0.3 * m.sum()) else -1
        if runs and runs[-1].pc == r:
            runs[-1].dur += step
        else:
            runs.append(RootRun(round(i * step, 3), r, step))
    return [r for r in runs if r.pc >= 0 and r.dur >= min_dur]


def root_changes(runs: list[RootRun]) -> list[float]:
    out: list[float] = []
    prev = None
    for r in runs:
        if r.pc != prev:
            out.append(r.start); prev = r.pc
    return out
