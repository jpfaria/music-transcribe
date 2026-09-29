import pytest
from music_transcribe.render.tab import token, bar_tab, bars_for
from music_transcribe.schema import Note, Harmony


def n(start, pitch, fret, string, art="", dur=0.3):
    return Note(start, start + dur, pitch, 0.8, articulation=art, string=string, fret=fret)


def test_tokens():
    assert token(n(0, 70, 6, 0)) == "6"
    assert token(n(0, 70, 6, 0, "bend:2")) == "6b8"
    assert token(n(0, 70, 6, 0, "bend:1")) == "6b7"
    assert token(n(0, 70, 6, 0, "bend-release")) == "6br"
    assert token(n(0, 70, 6, 0, "vibrato")) == "6~"
    assert token(n(0, 70, 6, 0, "slide:73")) == "6/9"
    assert token(n(0, 70, 6, 0, "hammer")) == "h6"
    assert token(n(0, 70, 6, 0, "pull")) == "p6"


def test_bar_tab_12_8_layout():
    bar = 4.8; notes = [n(0.0, 70, 6, 0), n(1.2, 65, 6, 1, "vibrato"), n(2.4, 58, 8, 3)]
    s = bar_tab(notes, 0.0, bar, "12/8", "guitar")
    lines = s.splitlines()
    assert len(lines) == 6 and lines[0].startswith("e|") and lines[5].startswith("E|")
    assert lines[0].count("|") == 6            # name sep + 4 beats + closing
    assert "6--" in lines[0] and "6~-" in lines[1] and "8--" in lines[3]


def test_bar_tab_4_4_has_16_cells():
    s = bar_tab([n(0.0, 70, 6, 0)], 0.0, 2.0, "4/4", "guitar")
    assert len(s.splitlines()[0]) == 2 + 16 * 3 + 5


def test_bar_tab_no_truncation_widens_cells():
    s = bar_tab([n(0.0, 70, 12, 0, "bend:2")], 0.0, 2.0, "4/4", "guitar")
    lines = s.splitlines()
    assert "12b14" in lines[0]
    widths = {len(l) for l in lines}
    assert len(widths) == 1


def test_bar_tab_collision_moves_to_next_cell():
    notes = [n(0.0, 70, 6, 0), n(0.05, 65, 9, 0)]
    s = bar_tab(notes, 0.0, 2.0, "4/4", "guitar")
    lines = s.splitlines()
    assert "6--9--" in lines[0]


def test_bars_for_rejects_nonpositive_bar_len():
    h = Harmony(120, "4/4", "C", 0.0, 0.0)
    with pytest.raises(ValueError):
        bars_for([n(0.0, 70, 6, 0)], h, "guitar")
