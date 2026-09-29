from __future__ import annotations
from music_transcribe.schema import Note


def score(note: Note, voiced_frac: float, repeated_low_run: bool) -> tuple[str, str]:
    if repeated_low_run:
        return "low", "bleed"
    if note.amplitude < 0.35 or voiced_frac < 0.2:
        return "low", "weak" if note.amplitude < 0.35 else "unstable"
    if note.amplitude < 0.6:
        return "medium", "weak"
    if voiced_frac < 0.5:
        return "medium", "unstable"
    return "high", ""


def mark_bleed_runs(notes: list[Note], low_pitch_max: int = 52, min_run: int = 6, max_gap: float = 0.5) -> set[int]:
    flagged: set[int] = set()
    run: list[int] = []
    for i, n in enumerate(notes):
        if run and n.pitch == notes[run[-1]].pitch and n.pitch <= low_pitch_max and n.start - notes[run[-1]].end < max_gap:
            run.append(i)
        else:
            if len(run) >= min_run:
                flagged.update(run)
            run = [i] if n.pitch <= low_pitch_max else []
    if len(run) >= min_run:
        flagged.update(run)
    return flagged
