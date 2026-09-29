import json
from pathlib import Path

from music_transcribe.cache import OutDir
from music_transcribe.render import render_all


def test_render_all_with_no_notes(tmp_path: Path):
    out = OutDir(tmp_path / "out")
    harmony = {
        "bpm": 52, "meter": "12/8", "key": "Bbm", "bar0": 0.75, "bar_len": 4.64,
        "chords": [{"bar": 1, "start": 0.75, "name": "Bbm", "root": "Bb", "quality": "min", "bass": "Bb", "confidence": 0.4}],
        "loop": ["Bbm"],
    }
    (out.stage("harmony") / "harmony.json").write_text(json.dumps(harmony))
    out.mark_done("tags")

    files = render_all(out)

    assert files["html"].exists()
    assert files["fragment"].exists()
    assert files["txt"].exists()
    assert "guitar" not in files and "bass" not in files and "piano" not in files
    assert out.done("render")

    html = files["html"].read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "Instrumental" in html
