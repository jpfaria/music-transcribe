from typer.testing import CliRunner
from music_transcribe.cli import app
from music_transcribe import __version__

runner = CliRunner()


def test_version():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0
    assert __version__ in r.output


from pathlib import Path
from music_transcribe import pipeline
from music_transcribe.schema import Tags, Harmony, Chord, LyricLine, Note


def test_run_skips_done_stages_and_renders(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"
    calls = []
    monkeypatch.setattr(pipeline, "read_tags", lambda p, **k: calls.append("tags") or Tags(title="T", artist="A"))
    monkeypatch.setattr(pipeline, "separate", lambda a, o, **k: calls.append("stems") or {s: (Path(o) / f"{s}.wav") for s in ["vocals", "guitar", "bass", "drums", "piano", "other"]})
    monkeypatch.setattr(pipeline, "to_wav", lambda s, d, **k: Path(d).write_bytes(b"RIFF") or Path(d))
    monkeypatch.setattr(pipeline, "ensure_whisper_model", lambda n, confirm, **k: Path("/m.bin"))
    monkeypatch.setattr(pipeline, "transcribe", lambda w, m, **k: calls.append("lyrics") or [LyricLine(1, 2, "hi", 0.9)])
    monkeypatch.setattr(pipeline, "analyze", lambda stems, **k: calls.append("harmony") or Harmony(100, "4/4", "C", 0.0, 2.4, [Chord(1, 0.0, "C", "C", "maj", "C", 0.5)], []))
    monkeypatch.setattr(pipeline, "transcribe_instrument", lambda w, inst, h, **k: calls.append(f"notes:{inst}") or [Note(0.1, 0.4, 60, 0.9, string=1, fret=1)])
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    assert r.exit_code == 0, r.output
    assert (out / "render" / "cifra.html").exists() and (out / "render" / "guitar.mid").exists()
    assert calls == ["tags", "stems", "lyrics", "harmony", "notes:guitar"]
    calls.clear()
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    assert r.exit_code == 0 and calls == []          # everything cached
    r = runner.invoke(app, ["harmony", str(audio), "--out", str(out), "--force"])
    assert r.exit_code == 0 and calls == ["harmony"]


def test_notes_rejects_unknown_instrument(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "separate", lambda a, o, **k: {s: (Path(o) / f"{s}.wav") for s in ["vocals", "guitar", "bass", "drums", "piano", "other"]})
    monkeypatch.setattr(pipeline, "to_wav", lambda s, d, **k: Path(d).write_bytes(b"RIFF") or Path(d))
    r = runner.invoke(app, ["notes", str(audio), "--out", str(out), "--instruments", "synth"])
    assert r.exit_code == 1
    assert "instrumento desconhecido" in r.output


def test_lyrics_fallback_no_double_prompt_when_model_matches_fallback(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "read_tags", lambda p, **k: Tags(title="T", artist="A"))
    monkeypatch.setattr(pipeline, "separate", lambda a, o, **k: {s: (Path(o) / f"{s}.wav") for s in ["vocals", "guitar", "bass", "drums", "piano", "other"]})
    monkeypatch.setattr(pipeline, "to_wav", lambda s, d, **k: Path(d).write_bytes(b"RIFF") or Path(d))
    count = {"n": 0}

    def fake_ensure(name, confirm, **k):
        count["n"] += 1
        return None
    monkeypatch.setattr(pipeline, "ensure_whisper_model", fake_ensure)
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--model", "medium.en",
                            "--language", "en", "--instruments", "guitar"])
    assert r.exit_code == 1
    assert "nenhum modelo Whisper" in r.output
    assert count["n"] == 1


def test_run_reports_missing_tool_with_brew_hint(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"

    def boom(p, **k):
        raise FileNotFoundError("ffprobe")
    monkeypatch.setattr(pipeline, "read_tags", boom)
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes"])
    assert r.exit_code == 1
    assert "brew install" in r.output


def test_run_passes_language_through_to_transcribe(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"
    seen = {}
    monkeypatch.setattr(pipeline, "read_tags", lambda p, **k: Tags(title="T", artist="A"))
    monkeypatch.setattr(pipeline, "separate", lambda a, o, **k: {s: (Path(o) / f"{s}.wav") for s in ["vocals", "guitar", "bass", "drums", "piano", "other"]})
    monkeypatch.setattr(pipeline, "to_wav", lambda s, d, **k: Path(d).write_bytes(b"RIFF") or Path(d))
    monkeypatch.setattr(pipeline, "ensure_whisper_model", lambda n, confirm, **k: Path("/m.bin"))

    def fake_transcribe(w, m, **k):
        seen["language"] = k.get("language")
        return [LyricLine(1, 2, "oi", 0.9)]
    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "analyze", lambda stems, **k: Harmony(100, "4/4", "C", 0.0, 2.4, [], []))
    monkeypatch.setattr(pipeline, "transcribe_instrument", lambda w, inst, h, **k: [Note(0.1, 0.4, 60, 0.9, string=1, fret=1)])
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--language", "pt", "--instruments", "guitar"])
    assert r.exit_code == 0, r.output
    assert seen["language"] == "pt"
