from __future__ import annotations
import librosa
import numpy as np


def _tempo_candidates(y: np.ndarray, sr: int) -> list[float]:
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
    ac = tg.mean(axis=1)
    bpms = librosa.tempo_frequencies(len(ac), sr=sr)
    idx = [i for i in np.argsort(ac)[::-1] if 30 < bpms[i] < 240][:8]
    return [float(bpms[i]) for i in idx]


def _meter(y: np.ndarray, sr: int, bpm: float) -> str:
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
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
    cands = _tempo_candidates(y, sr)
    # expand each candidate by integer factors so octave errors are recoverable
    expanded = sorted({round(c / f, 2) for c in cands for f in (1, 2, 3, 4)} | {round(c * 2, 2) for c in cands})
    expanded = [b for b in expanded if 30 <= b <= 240]
    if len(root_changes) >= 3:
        target_bar = float(np.median(np.diff(root_changes)))
        def err(b: float) -> float:
            bar = 4 * 60.0 / b
            # allow one chord per bar or per two bars
            return min(abs(bar - target_bar), abs(2 * bar - target_bar)) / target_bar
        bpm = min(expanded, key=err)
    else:
        bpm = cands[0]
    return round(bpm, 1), _meter(y, sr, bpm)
