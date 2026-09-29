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


def raw_notes(wav: Path, predictor=None, min_amp: float = 0.45) -> list[Note]:
    events = (predictor or _basic_pitch_predict)(wav)
    events = sorted((e for e in events if e[3] >= min_amp), key=lambda e: e[0])
    out: list[Note] = []
    for s, e, p, a in events:
        louder_overlap = any(abs(o[0] - s) < 0.06 and o[3] > a for o in events)
        if not louder_overlap:
            out.append(Note(round(s, 3), round(e, 3), p, round(a, 3)))
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
