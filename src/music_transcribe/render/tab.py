from __future__ import annotations
from music_transcribe.schema import Note, Harmony

STRING_NAMES = {"guitar": "eBGDAE", "bass": "GDAE"}
CELL = 3


def token(n: Note) -> str:
    f = "" if n.fret is None else str(n.fret)
    a = n.articulation
    if a.startswith("bend:") and n.fret is not None:
        return f"{f}b{n.fret + int(a.split(':')[1])}"
    if a == "bend-release":
        return f"{f}br"
    if a == "vibrato":
        return f"{f}~"
    if a.startswith("slide:") and n.fret is not None and n.string is not None:
        # target fret on the same string
        return f"{f}/{n.fret + (int(a.split(':')[1]) - n.pitch)}"
    if a == "hammer":
        return f"h{f}"
    if a == "pull":
        return f"p{f}"
    return f


def bar_tab(notes: list[Note], bar_start: float, bar_len: float, meter: str, instrument: str) -> str:
    names = STRING_NAMES[instrument]
    cells = 12 if meter == "12/8" else 16
    per_beat = cells // 4
    sub = bar_len / cells
    in_bar = [n for n in notes if n.string is not None and n.fret is not None
              and bar_start <= n.start < bar_start + bar_len]
    cell_w = max(CELL, max((len(token(n)) for n in in_bar), default=0) + 1)
    empty = "-" * cell_w
    grid: list[list[str]] = [[empty] * cells for _ in names]
    dropped = 0
    for n in sorted(in_bar, key=lambda x: x.start):
        k0 = min(cells - 1, int(round((n.start - bar_start) / sub)))
        t = token(n).ljust(cell_w, "-")
        k = k0
        while k < cells and grid[n.string][k] != empty:
            k += 1
        if k < cells:
            grid[n.string][k] = t
        else:
            dropped += 1
    lines = []
    for s, name in enumerate(names):
        row = "".join(("|" if k % per_beat == 0 else "") + grid[s][k] for k in range(cells))
        lines.append(f"{name}|{row}|")
    if dropped:
        lines.append(f"  * {dropped} nota(s) omitida(s) por colisão")
    return "\n".join(lines)


def bars_for(notes: list[Note], harmony: Harmony, instrument: str) -> list[tuple[int, str]]:
    if harmony.bar_len <= 0:
        raise ValueError("bar_len must be positive")
    if not notes:
        return []
    last = max(n.start for n in notes)
    out = []
    b = 0
    while harmony.bar0 + b * harmony.bar_len <= last:
        t0 = harmony.bar0 + b * harmony.bar_len
        in_bar = [n for n in notes if t0 <= n.start < t0 + harmony.bar_len and n.fret is not None]
        if in_bar:
            out.append((b + 1, bar_tab(in_bar, t0, harmony.bar_len, harmony.meter, instrument)))
        b += 1
    return out
