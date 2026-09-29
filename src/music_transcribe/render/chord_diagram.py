from __future__ import annotations
import re
from dataclasses import dataclass
from html import escape
from music_transcribe.schema import NOTE_NAMES
from music_transcribe.harmony.chords import TEMPLATES, SUFFIX

OPEN = [40, 45, 50, 55, 59, 64]  # low E → high e
_SUFFIX_TO_Q = {v: k for k, v in SUFFIX.items()}

# movable shapes: offsets from the root fret; root string index (0 = low E) given by key
# E-shapes (root on string 0) and A-shapes (root on string 1). -1 = mute.
E_SHAPES = {"maj": [0, 2, 2, 1, 0, 0], "min": [0, 2, 2, 0, 0, 0], "7": [0, 2, 0, 1, 0, 0], "m7": [0, 2, 0, 0, 0, 0],
            "maj7": [0, -1, 1, 1, 0, -1], "sus4": [0, 2, 2, 2, 0, 0], "sus2": [0, 2, 4, 4, 0, 0]}
A_SHAPES = {"maj": [-1, 0, 2, 2, 2, 0], "min": [-1, 0, 2, 2, 1, 0], "7": [-1, 0, 2, 0, 2, 0], "m7": [-1, 0, 2, 0, 1, 0],
            "maj7": [-1, 0, 2, 1, 2, 0], "sus4": [-1, 0, 2, 2, 3, 0], "sus2": [-1, 0, 2, 2, 0, 0],
            "9": [-1, 0, 2, 4, 2, 3], "add9": [-1, 0, 2, 4, 2, 0], "dim": [-1, 0, 1, 2, 1, -1], "m7b5": [-1, 0, 1, 0, 1, -1]}


@dataclass
class Voicing:
    frets: list[int]
    base: int
    barre: tuple[int, int, int] | None = None  # (fret, from_string, to_string)


def parse_chord(name: str) -> tuple[int, str, int | None]:
    m = re.match(r"^([A-G][b#]?)(.*?)(?:/([A-G][b#]?))?$", name.strip())
    if not m:
        raise ValueError(name)
    root, suf, bass = m.groups()
    if root not in NOTE_NAMES and "#" in root:
        root_pc = (NOTE_NAMES.index(root[0]) + 1) % 12
    else:
        root_pc = NOTE_NAMES.index(root)
    q = _SUFFIX_TO_Q.get(suf)
    if q is None:
        raise ValueError(f"unknown quality '{suf}' in {name}")
    bass_pc = None
    if bass:
        bass_pc = (NOTE_NAMES.index(bass[0]) + 1) % 12 if "#" in bass else NOTE_NAMES.index(bass)
    return root_pc, q, bass_pc


def _valid(v: Voicing | None, tones: set[int]) -> bool:
    """A voicing is usable when every chord tone still sounds and the
    fretted (non-open, non-muted) notes fit inside a 5-fret diagram window."""
    if v is None:
        return False
    sounding = [f for f in v.frets if f >= 0]
    if not sounding:
        return False
    got = {(o + f) % 12 for o, f in zip(OPEN, v.frets) if f >= 0}
    if not (tones <= got):
        return False
    nonzero = [f for f in v.frets if f > 0]
    if nonzero and max(nonzero) - min(nonzero) > 4:
        return False
    return True


def _shape(root_pc: int, q: str) -> Voicing | None:
    tones = {(root_pc + iv) % 12 for iv in TEMPLATES[q]}
    options = []
    for shapes, root_string in ((E_SHAPES, 0), (A_SHAPES, 1)):
        if q not in shapes:
            continue
        root_fret = (root_pc - OPEN[root_string]) % 12
        for base in (root_fret, root_fret + 12):
            if base > 12:
                continue
            frets = [f if f < 0 else f + base for f in shapes[q]]
            if max(frets) <= 15:
                options.append((base, frets, root_string))
    if not options:
        return None
    base, frets, rs = min(options, key=lambda o: o[0])
    barre = None
    if base > 0:
        played = [i for i, f in enumerate(frets) if f >= 0]
        if sum(1 for i in played if frets[i] == base) >= 2:
            barre = (base, played[0], played[-1])
    v = Voicing(frets, base if base > 0 else 1, barre)
    return v if _valid(v, tones) else None


def _fallback(root_pc: int, q: str) -> Voicing | None:
    tones = {(root_pc + iv) % 12 for iv in TEMPLATES[q]}
    best = None
    for base in range(1, 10):
        frets = []
        for o in OPEN:
            pick = -1
            for f in range(base, base + 4):
                if (o + f) % 12 in tones:
                    pick = f; break
            frets.append(pick)
        got = {(o + f) % 12 for o, f in zip(OPEN, frets) if f >= 0}
        if tones <= got:
            n_mute = frets.count(-1)
            if best is None or n_mute < best[0]:
                best = (n_mute, Voicing(frets, base))
    v = best[1] if best else None
    return v if _valid(v, tones) else None


def _base_of(frets: list[int]) -> int:
    """Fret shown at the top-left of the diagram: the lowest fretted (non-open) note, or 1 in open position."""
    nonzero = [f for f in frets if f > 0]
    return min(nonzero) if nonzero else 1


def _with_bass(v: Voicing, bass_pc: int) -> Voicing:
    """Candidate voicing with the bass note seated on the lowest reachable string, muting the strings below it."""
    lo = min(range(1, 3), key=lambda s: abs(((bass_pc - OPEN[s]) % 12) - v.base))
    f = (bass_pc - OPEN[lo]) % 12
    new = list(v.frets)
    for s in range(lo):
        new[s] = -1
    new[lo] = f
    return Voicing(new, _base_of(new), None)


def voicing_for(name: str) -> Voicing | None:
    try:
        root_pc, q, bass_pc = parse_chord(name)
    except ValueError:
        return None
    tones = {(root_pc + iv) % 12 for iv in TEMPLATES[q]}
    v = _shape(root_pc, q) or _fallback(root_pc, q)
    if v is None:
        return None
    if bass_pc is not None and bass_pc != root_pc:
        candidate = _with_bass(v, bass_pc)
        lowest_pc = next(((o + f) % 12 for o, f in zip(OPEN, candidate.frets) if f >= 0), None)
        if _valid(candidate, tones) and lowest_pc == bass_pc:
            return candidate
        # inversion isn't reachable cleanly: a correct root-position diagram
        # beats an invalid or misleading slash-bass one.
    return v


def svg(v: Voicing, name: str) -> str:
    W, H, x0, y0, sy, n = 120, 150, 22, 34, 20, 5
    sx = (W - 2 * x0) / 5
    base = v.base if v.base > 1 else 1
    o = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(name)}">']
    for i in range(6):
        x = x0 + i * sx
        o.append(f'<line class="st" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + n * sy}" stroke-width="1"/>')
    for j in range(n + 1):
        y = y0 + j * sy
        cls = "nut" if (j == 0 and base == 1) else "st"
        o.append(f'<line class="{cls}" x1="{x0}" y1="{y}" x2="{x0 + 5 * sx:.1f}" y2="{y}" stroke-width="1"/>')
    if base > 1:
        o.append(f'<text class="lbl" x="{x0 - 16}" y="{y0 + sy / 2 + 4}" text-anchor="middle">{base}</text>')
    if v.barre:
        fr, a, b = v.barre
        y = y0 + (fr - base + 0.5) * sy
        o.append(f'<rect class="brr" x="{x0 + a * sx - 7:.1f}" y="{y - 7:.1f}" width="{(b - a) * sx + 14:.1f}" height="14" rx="7"/>')
    for i, f in enumerate(v.frets):
        x = x0 + i * sx
        if f == -1:
            o.append(f'<text class="lbl" x="{x:.1f}" y="{y0 - 8}" text-anchor="middle">×</text>')
        elif f == 0:
            o.append(f'<circle cx="{x:.1f}" cy="{y0 - 11}" r="4" fill="none" class="st" stroke-width="1"/>')
        else:
            y = y0 + (f - base + 0.5) * sy
            o.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="7"/>')
    o.append("</svg>")
    return "".join(o)
