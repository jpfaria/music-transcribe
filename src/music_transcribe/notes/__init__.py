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


def _contour_drift(cents: np.ndarray) -> float:
    """Median cents of the last third minus the first third of the voiced contour.

    0.0 when there isn't enough voiced signal to say anything about direction.
    """
    c = np.asarray(cents, dtype=float)
    ok = ~np.isnan(c)
    if ok.sum() < 6:
        return 0.0
    c = c[ok]
    third = max(2, len(c) // 3)
    head, tail = np.nanmedian(c[:third]), np.nanmedian(c[-third:])
    drift = tail - head
    return float(drift) if np.isfinite(drift) else 0.0


def transcribe_instrument(wav: Path, instrument: str, harmony: Harmony, predictor=None) -> list[Note]:
    import librosa
    y, sr = load_mono(wav, 22050)
    notes = raw_notes(wav, predictor=predictor)
    bleed = mark_bleed_runs(notes)
    hop_s = 256 / sr
    drifts: list[float] = []
    for i, n in enumerate(notes):
        cents, voiced = f0_contour(y, sr, n)
        n.articulation = classify_contour(cents, hop_s)
        n.confidence, n.reason = score(n, voiced, i in bleed)
        drifts.append(_contour_drift(cents))
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    for i in range(1, len(notes)):
        p, c = notes[i - 1], notes[i]
        gap = c.start - p.end
        if gap > 0.15 or p.articulation.startswith("bend"):
            continue
        if gap <= 0:
            # Touching/overlapping notes: no silent gap to measure voicing in, so
            # let onset strength alone decide hammer/pull. A slide across a
            # touching pair instead shows up as the previous note's own pitch
            # drifting toward the next note's pitch right before the join.
            drift = drifts[i - 1]
            pitch_diff = c.pitch - p.pitch
            if pitch_diff != 0 and drift != 0 and (drift > 0) == (pitch_diff > 0) and abs(drift) >= 50:
                p.articulation = p.articulation or f"slide:{c.pitch}"
                continue
            art = detect_legato(p, c, 0.0, onset_ratio(y, sr, c.start, oenv=oenv))
        else:
            art = detect_legato(p, c, _gap_voiced(y, sr, p.end, c.start), onset_ratio(y, sr, c.start, oenv=oenv))
        if art.startswith("slide"):
            p.articulation = p.articulation or art
        elif art:
            c.articulation = c.articulation or art
    if instrument in FRETTED:
        assign_positions(notes, TUNINGS[instrument], preferred=preferred_region(harmony.key) if instrument == "guitar" else (0, 12))
    return notes
