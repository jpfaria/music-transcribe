import pytest
from music_transcribe.render.chord_diagram import voicing_for, svg, parse_chord
from music_transcribe.harmony.chords import TEMPLATES
from music_transcribe.schema import NOTE_NAMES

OPEN = [40, 45, 50, 55, 59, 64]


def pcs(v):
    return {(o + f) % 12 for o, f in zip(OPEN, v.frets) if f >= 0}


def test_parse():
    assert parse_chord("Bbm") == (10, "min", None)
    assert parse_chord("Db/F") == (1, "maj", 5)
    assert parse_chord("Ebm7b5") == (3, "m7b5", None)
    assert parse_chord("F7") == (5, "7", None)


@pytest.mark.parametrize("name", ["Bbm", "Db", "Ebm", "F", "F7", "Ebm7b5", "Dbmaj7", "Absus4", "Bb9", "Gdim", "Db/F"])
def test_voicing_covers_chord_tones(name):
    v = voicing_for(name)
    assert v is not None, name
    root, q, _ = parse_chord(name)
    want = {(root + iv) % 12 for iv in TEMPLATES[q]}
    assert want <= pcs(v), (name, v)
    assert 0 <= v.base <= 12 and all(f == -1 or 0 <= f - (v.base - 1 if v.base > 1 else 0) <= 5 for f in v.frets)


def test_svg_has_dots_and_label():
    v = voicing_for("Bbm")
    s = svg(v, "Bbm")
    assert s.startswith("<svg") and "<circle" in s and "Bbm" in s
