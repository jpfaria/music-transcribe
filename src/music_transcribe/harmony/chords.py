from __future__ import annotations
import numpy as np
from music_transcribe.schema import Chord, NOTE_NAMES

TEMPLATES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
    "dim": (0, 3, 6), "m7b5": (0, 3, 6, 10), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
    "add9": (0, 4, 7, 2), "9": (0, 4, 7, 10, 2),
}
SUFFIX = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "m7": "m7", "dim": "dim", "m7b5": "m7b5",
          "sus2": "sus2", "sus4": "sus4", "add9": "add9", "9": "9"}
# tetrads/pentads need a small bonus to beat their triad subset only when the extra tone is really present
_COMPLEXITY_PENALTY = {k: 0.04 * (len(v) - 3) for k, v in TEMPLATES.items()}


def _vec(root: int, quality: str) -> np.ndarray:
    v = np.zeros(12)
    for iv in TEMPLATES[quality]:
        v[(root + iv) % 12] = 1.0
    return v / np.linalg.norm(v)


def chord_for_bar(chroma: np.ndarray, bass_pc: int | None, bar: int, start: float, bass_weight: float = 0.6) -> Chord:
    c = np.asarray(chroma, dtype=float)
    c = c / (c.max() + 1e-9)
    if bass_pc is not None:
        b = np.zeros(12); b[bass_pc % 12] = 1.0
        c = c + bass_weight * b
    c = c / (np.linalg.norm(c) + 1e-9)
    scored = []
    for root in range(12):
        for q in TEMPLATES:
            s = float(np.dot(_vec(root, q), c)) - _COMPLEXITY_PENALTY[q]
            if bass_pc is not None and (bass_pc - root) % 12 not in TEMPLATES[q]:
                s -= 0.15  # bass note not a chord tone: unlikely
            scored.append((s, root, q))
    scored.sort(reverse=True)
    (s1, root, q), s2 = scored[0], scored[1][0]
    name = NOTE_NAMES[root] + SUFFIX[q]
    bass_name = NOTE_NAMES[bass_pc % 12] if bass_pc is not None else NOTE_NAMES[root]
    if bass_pc is not None and bass_pc % 12 != root:
        name += "/" + bass_name
    return Chord(bar=bar, start=float(start), name=name, root=NOTE_NAMES[root], quality=q,
                 bass=bass_name, confidence=round(max(0.0, min(1.0, (s1 - s2) * 4)), 3))


def detect_loop(names: list[str], max_period: int = 8, min_match: float = 0.8) -> list[str]:
    n = len(names)
    for p in range(2, max_period + 1):
        if n < 2 * p:
            break
        pairs = [(names[i], names[i + p]) for i in range(n - p)]
        match = sum(a == b for a, b in pairs) / len(pairs)
        if match >= min_match:
            # majority vote per position
            out = []
            for k in range(p):
                col = names[k::p]
                out.append(max(set(col), key=col.count))
            return out
    return []


_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chroma_mean: np.ndarray) -> str:
    prof = np.asarray(chroma_mean, dtype=float)
    best = None
    for i in range(12):
        for name, tmpl in (("", _MAJ), ("m", _MIN)):
            r = np.corrcoef(np.roll(tmpl, i), prof)[0, 1]
            if best is None or r > best[0]:
                best = (r, NOTE_NAMES[i] + name)
    return best[1]
