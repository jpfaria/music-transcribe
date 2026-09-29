from __future__ import annotations
from pathlib import Path
import librosa
import numpy as np
from music_transcribe.audio import load_mono
from music_transcribe.schema import Harmony, Chord
from music_transcribe.harmony.bassroot import bass_roots, root_changes
from music_transcribe.harmony.tempo import estimate_tempo
from music_transcribe.harmony.grid import fit_grid, refine_bar_len
from music_transcribe.harmony.chords import chord_for_bar, detect_loop, estimate_key

HARMONIC = ["bass", "piano", "guitar", "other"]


def analyze(stems: dict[str, Path], sr: int = 22050, mix: Path | None = None) -> Harmony:
    parts = {k: load_mono(stems[k], sr)[0] for k in HARMONIC if k in stems}
    if not parts:
        raise RuntimeError("no harmonic stems found (need at least one of bass, piano, guitar, other)")
    bass = parts.get("bass")
    n = max(len(v) for v in parts.values())
    h = np.zeros(n, dtype=np.float32)
    for v in parts.values():
        h[: len(v)] += v
    if mix:
        full = load_mono(mix, sr)[0]
    elif "drums" in stems:
        drums = load_mono(stems["drums"], sr)[0]
        full_n = max(n, len(drums))
        full = np.zeros(full_n, dtype=np.float32)
        full[: len(h)] += h
        full[: len(drums)] += drums
    else:
        full = h
    runs = bass_roots(bass, sr) if bass is not None else []
    changes = root_changes(runs)
    bpm, meter = estimate_tempo(full, sr, changes)
    bar_len0 = 4 * 60.0 / bpm
    bar_len = refine_bar_len(changes, bar_len0)
    if bar_len != bar_len0:
        bpm = round(240.0 / bar_len, 1)
    bar0 = fit_grid(changes, bar_len)
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=h, sr=sr, hop_length=hop)
    key = estimate_key(chroma.mean(axis=1))
    # bass pc per frame from runs
    t_frames = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    bass_pc_frame = np.full(len(t_frames), -1)
    for r in runs:
        bass_pc_frame[(t_frames >= r.start) & (t_frames < r.start + r.dur)] = r.pc
    total = len(h) / sr
    chords: list[Chord] = []
    b = 0
    while bar0 + b * bar_len < total:
        t0, t1 = bar0 + b * bar_len, bar0 + (b + 1) * bar_len
        m = (t_frames >= max(0.0, t0)) & (t_frames < t1)
        if m.sum() == 0:
            b += 1; continue
        c = chroma[:, m].mean(axis=1)
        bpcs = bass_pc_frame[m]; bpcs = bpcs[bpcs >= 0]
        bass_pc = int(np.bincount(bpcs, minlength=12).argmax()) if len(bpcs) else None
        chords.append(chord_for_bar(c, bass_pc, bar=b + 1, start=round(t0, 3)))
        b += 1
    return Harmony(bpm=bpm, meter=meter, key=key, bar0=bar0, bar_len=round(bar_len, 4),
                   chords=chords, loop=detect_loop([c.name for c in chords]))
