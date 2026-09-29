from music_transcribe.render.text import cifra_txt
from music_transcribe.schema import Tags, Harmony, Chord, LyricLine, Note


def test_cifra_txt_sections():
    h = Harmony(52, "12/8", "Bbm", 0.75, 4.64, [Chord(1, 0.75, "Bbm", "Bb", "min", "Bb", 0.4)], ["Bbm", "Db", "Ebm", "Bbm"])
    txt = cifra_txt(Tags(title="T", artist="A"), h, [LyricLine(20, 25, "Moonlight", 0.9)], {"guitar": [Note(1.0, 1.3, 70, 0.8, string=0, fret=6)]})
    assert "T — A" in txt and "Bbm" in txt and "Moonlight" in txt and "e|" in txt and "12/8" in txt


def test_txt_clamps_negative_bar_start():
    h = Harmony(52, "12/8", "Bbm", -3.89, 4.64, [Chord(1, -3.89, "Bbm", "Bb", "min", "Bb", 0.4)], [])
    txt = cifra_txt(Tags(), h, [], {})
    assert "0:00" in txt and "-" not in txt.split("ACORDES POR COMPASSO")[1].splitlines()[1]


def test_txt_scale_and_legend():
    h = Harmony(52, "12/8", "Bbm", 0.0, 4.64, [], [])
    txt = cifra_txt(Tags(), h, [], {"guitar": [Note(1.0, 1.3, 70, 0.8, confidence="low", string=0, fret=6)]})
    assert "ESCALA DO SOLO" in txt and "Bb Db Eb F Ab" in txt and "casa 6" in txt
    assert "(6)" in txt and "(6)=confiança baixa" in txt
