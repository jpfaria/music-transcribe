from music_transcribe.render.text import cifra_txt
from music_transcribe.schema import Tags, Harmony, Chord, LyricLine, Note


def test_cifra_txt_sections():
    h = Harmony(52, "12/8", "Bbm", 0.75, 4.64, [Chord(1, 0.75, "Bbm", "Bb", "min", "Bb", 0.4)], ["Bbm", "Db", "Ebm", "Bbm"])
    txt = cifra_txt(Tags(title="T", artist="A"), h, [LyricLine(20, 25, "Moonlight", 0.9)], {"guitar": [Note(1.0, 1.3, 70, 0.8, string=0, fret=6)]})
    assert "T — A" in txt and "Bbm" in txt and "Moonlight" in txt and "e|" in txt and "12/8" in txt
