from music_transcribe.render.html import cifra_html
from music_transcribe.schema import Tags, Harmony, Chord, LyricLine, Note

H = Harmony(52, "12/8", "Bbm", 0.75, 4.64,
            [Chord(1, 0.75, "Bbm", "Bb", "min", "Bb", 0.4), Chord(2, 5.39, "Db", "Db", "maj", "Db", 0.1)],
            ["Bbm", "Db", "Ebm", "Bbm"])


def test_html_sections_and_confidence_markers():
    html = cifra_html(Tags(title="Midnight", artist="CBR"), H,
                      [LyricLine(20, 25, "Moonlight", 0.9), LyricLine(25, 30, "For another glance", 0.4)],
                      {"guitar": [Note(1.0, 1.3, 70, 0.8, articulation="bend:2", confidence="low", reason="bleed", string=0, fret=6)]})
    assert html.startswith("<title>Midnight</title>")
    assert 'class="chords-grid"' in html and "<svg" in html
    assert "Moonlight" in html and '"line low"' in html            # low-confidence lyric line marked
    assert "6b8" in html and "conf-low" in html                      # tab token and note confidence class
    assert "prefers-color-scheme: dark" in html and 'data-theme="dark"' in html


def test_html_instrumental():
    html = cifra_html(Tags(), H, [], {})
    assert "instrumental" in html.lower()
