import numpy as np
from music_transcribe.harmony.chords import chord_for_bar, detect_loop, estimate_key, TEMPLATES


def chroma(pcs, w=1.0):
    c = np.zeros(12)
    for p in pcs: c[p % 12] = w
    return c + 0.05


def test_minor_triad_named():
    ch = chord_for_bar(chroma([10, 1, 5]), bass_pc=10, bar=1, start=0.0)
    assert ch.name == "Bbm" and ch.quality == "min" and ch.confidence > 0


def test_dominant_seventh():
    ch = chord_for_bar(chroma([5, 9, 0, 3]), bass_pc=5, bar=1, start=0.0)
    assert ch.name == "F7"


def test_inversion_uses_slash():
    ch = chord_for_bar(chroma([1, 5, 8]), bass_pc=5, bar=1, start=0.0)
    assert ch.name == "Db/F"


def test_half_diminished():
    ch = chord_for_bar(chroma([3, 6, 9, 1]), bass_pc=3, bar=1, start=0.0)
    assert ch.name == "Ebm7b5"


def test_root_position_beats_sus_reinterpretation():
    # Db major (Db-F-Ab) plus overtone leakage at Eb and C that could suggest Ab sus4.
    c = np.full(12, 0.05)
    c[1] = 1.0   # Db
    c[5] = 1.0   # F
    c[8] = 1.0   # Ab
    c[3] = 0.45  # Eb overtone leakage
    c[0] = 0.45  # C overtone leakage
    ch = chord_for_bar(c, bass_pc=1, bar=1, start=0.0)
    assert ch.name == "Db"


def test_detect_loop():
    seq = ["Bbm", "Db", "Ebm", "Bbm"] * 5 + ["Bbm", "Db"]
    assert detect_loop(seq) == ["Bbm", "Db", "Ebm", "Bbm"]
    assert detect_loop(["A", "B", "C", "D", "E", "F"]) == []


def test_estimate_key_minor():
    prof = np.zeros(12)
    for p, w in [(10, 3), (1, 2), (5, 2), (3, 1), (8, 1), (0, 0.5)]: prof[p] = w
    assert estimate_key(prof) == "Bbm"
