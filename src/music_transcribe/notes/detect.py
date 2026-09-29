from __future__ import annotations
from pathlib import Path
import librosa
import numpy as np
from music_transcribe.schema import Note


def _basic_pitch_predict(path: Path) -> list[tuple[float, float, int, float]]:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, _, events = predict(str(path), ICASSP_2022_MODEL_PATH, onset_threshold=0.5, frame_threshold=0.3,
                           minimum_note_length=80, minimum_frequency=30, maximum_frequency=2000, melodia_trick=True)
    return [(float(s), float(e), int(p), float(a)) for s, e, p, a, _ in events]


POLYPHONIC = {"piano"}


def raw_notes(wav: Path, predictor=None, min_amp: float = 0.3, instrument: str = "guitar") -> list[Note]:
    """basic-pitch notes above min_amp. Piano keeps simultaneous notes. Other instruments keep only
    the loudest of notes starting within 60 ms of each other; a kept note that won over at least one
    suppressed note is marked medium / "chord" (it was probably a chord, and the tab shows one note)."""
    events = (predictor or _basic_pitch_predict)(wav)
    events = sorted((e for e in events if e[3] >= min_amp), key=lambda e: e[0])
    if instrument in POLYPHONIC:
        return [Note(round(s, 3), round(e, 3), p, round(a, 3)) for s, e, p, a in events]
    out: list[Note] = []
    for s, e, p, a in events:
        near = [o for o in events if abs(o[0] - s) < 0.06 and o != (s, e, p, a)]
        if any(o[3] > a for o in near):
            continue
        n = Note(round(s, 3), round(e, 3), p, round(a, 3))
        if any(o[3] < a for o in near):
            n.confidence, n.reason = "medium", "chord"
        out.append(n)
    return out


def f0_contour(y: np.ndarray, sr: int, note: Note, hop: int = 256) -> tuple[np.ndarray, float]:
    a, b = int(note.start * sr), int(note.end * sr)
    seg = y[a:b]
    if len(seg) < hop * 4:
        return np.array([np.nan]), 0.0
    base = librosa.midi_to_hz(note.pitch)
    fmin, fmax = base * 2 ** (-7 / 12), base * 2 ** (7 / 12)
    f0, vf, _ = librosa.pyin(seg, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop)
    cents = 1200 * np.log2(f0 / base)
    cents[~vf] = np.nan
    return cents, float(vf.mean())
