from __future__ import annotations
from music_transcribe.schema import Note, NOTE_NAMES

TUNINGS: dict[str, list[int]] = {"guitar": [64, 59, 55, 50, 45, 40], "bass": [43, 38, 33, 28]}


def preferred_region(key: str) -> tuple[int, int]:
    root = key[:-1] if key.endswith("m") else key
    pc = NOTE_NAMES.index(root)
    fret = (pc - 4) % 12  # 6th string open E = pc 4
    return (max(0, fret - 2), fret + 4) if fret > 0 else (0, 4)


def _candidates(pitch: int, tuning: list[int], max_fret: int) -> list[tuple[int, int]]:
    return [(s, pitch - o) for s, o in enumerate(tuning) if 0 <= pitch - o <= max_fret]


def assign_positions(notes: list[Note], tuning: list[int], preferred: tuple[int, int] = (3, 15), max_fret: int = 22) -> None:
    prev_fret: int | None = None
    lo, hi = preferred
    for n in notes:
        cands = _candidates(n.pitch, tuning, max_fret)
        if not cands:
            n.string = n.fret = None
            continue

        def cost(c: tuple[int, int]) -> float:
            s, f = c
            jump = abs(f - prev_fret) if prev_fret is not None else abs(f - (lo + hi) / 2)
            region = 0 if lo <= f <= hi else min(abs(f - lo), abs(f - hi))
            open_pen = 0.5 if f == 0 and prev_fret and prev_fret > 3 else 0
            return jump + 1.5 * region + open_pen

        n.string, n.fret = min(cands, key=cost)
        prev_fret = n.fret
