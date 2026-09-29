from __future__ import annotations
from music_transcribe.schema import NOTE_NAMES
from music_transcribe.notes.fretboard import preferred_region

MINOR_PENT = [0, 3, 5, 7, 10]
MAJOR_PENT = [0, 2, 4, 7, 9]


def scale_for(key: str) -> dict:
    """Pentatonic to solo over `key` ("Bbm", "Db").

    Minor key → minor pentatonic. Major key → major pentatonic, which is the relative minor
    pentatonic started from the major root. box_fret is the root on the low E string, i.e. the
    centre of fretboard.preferred_region(key) (0 for E/Em).
    """
    minor = key.endswith("m")
    root = key[:-1] if minor else key
    pc = NOTE_NAMES.index(root)
    notes = [NOTE_NAMES[(pc + i) % 12] for i in (MINOR_PENT if minor else MAJOR_PENT)]
    if minor:
        name = f"{root} menor pentatônica"
    else:
        name = f"{root} maior pentatônica (= {NOTE_NAMES[(pc + 9) % 12]}m pentatônica)"
    _, hi = preferred_region(key)
    return {"name": name, "notes": notes, "box_fret": max(0, hi - 4)}
