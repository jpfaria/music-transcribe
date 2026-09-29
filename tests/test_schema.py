from pathlib import Path
from music_transcribe.schema import Note, Chord, Harmony, LyricLine, pitch_name, save_json, load_json


def test_pitch_name():
    assert pitch_name(58) == "Bb3"
    assert pitch_name(60) == "C4"


def test_roundtrip_list(tmp_path: Path):
    notes = [Note(0.0, 0.5, 58, 0.9, articulation="bend:2", confidence="low", reason="bleed", string=1, fret=6)]
    p = tmp_path / "n.json"
    save_json(notes, p)
    back = load_json(Note, p)
    assert back == notes


def test_roundtrip_nested(tmp_path: Path):
    h = Harmony(52.0, "12/8", "Bbm", 0.75, 4.64, [Chord(1, 0.75, "Bbm", "Bb", "min", "Bb", 0.3)], ["Bbm", "Db", "Ebm", "Bbm"])
    p = tmp_path / "h.json"
    save_json(h, p)
    assert load_json(Harmony, p) == h
