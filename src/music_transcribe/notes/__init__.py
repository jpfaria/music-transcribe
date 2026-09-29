from __future__ import annotations
from pathlib import Path
import numpy as np
from music_transcribe.audio import load_mono
from music_transcribe.schema import Note, Harmony
from music_transcribe.notes.detect import raw_notes, f0_contour
from music_transcribe.notes.articulation import classify_contour, detect_legato, onset_ratio
from music_transcribe.notes.confidence import score, mark_bleed_runs
from music_transcribe.notes.fretboard import assign_positions, preferred_region, TUNINGS

FRETTED = {"guitar", "bass"}


def _gap_voiced(y: np.ndarray, sr: int, a: float, b: float) -> float:
    import librosa
    seg = y[int(a * sr): int(b * sr)]
    if len(seg) < 2048:
        return 0.0
    _, vf, _ = librosa.pyin(seg, fmin=60, fmax=1500, sr=sr, frame_length=2048, hop_length=256)
    return float(vf.mean())


def transcribe_instrument(wav: Path, instrument: str, harmony: Harmony, predictor=None) -> list[Note]:
    y, sr = load_mono(wav, 22050)
    notes = raw_notes(wav, predictor=predictor)
    bleed = mark_bleed_runs(notes)
    hop_s = 256 / sr
    for i, n in enumerate(notes):
        cents, voiced = f0_contour(y, sr, n)
        n.articulation = classify_contour(cents, hop_s)
        n.confidence, n.reason = score(n, voiced, i in bleed)
    for i in range(1, len(notes)):
        p, c = notes[i - 1], notes[i]
        if c.start - p.end > 0.15 or p.articulation.startswith("bend"):
            continue
        art = detect_legato(p, c, _gap_voiced(y, sr, p.end, c.start) if c.start > p.end else 1.0,
                            onset_ratio(y, sr, c.start))
        if art.startswith("slide"):
            p.articulation = p.articulation or art
        elif art:
            c.articulation = c.articulation or art
    if instrument in FRETTED:
        assign_positions(notes, TUNINGS[instrument], preferred=preferred_region(harmony.key) if instrument == "guitar" else (0, 12))
    return notes
