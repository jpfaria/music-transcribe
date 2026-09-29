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
    monkeypatch.setattr(pipeline, "vocal_activity_for", lambda p, **k: [(0.0, 100.0)])
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
    calls.clear()
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    assert r.exit_code == 0 and calls == ["notes:guitar"]   # harmony --force invalidated notes


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
    monkeypatch.setattr(pipeline, "vocal_activity_for", lambda p, **k: [(0.0, 100.0)])
    count = {"n": 0}

    def fake_ensure(name, confirm, **k):
        count["n"] += 1
        return None
    monkeypatch.setattr(pipeline, "ensure_whisper_model", fake_ensure)
    monkeypatch.setattr(pipeline, "analyze", lambda stems, **k: Harmony(100, "4/4", "C", 0.0, 2.4, [], []))
    monkeypatch.setattr(pipeline, "transcribe_instrument", lambda w, inst, h, **k: [])
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--model", "medium.en",
                            "--language", "en", "--instruments", "guitar"])
    # a missing model no longer aborts `run` (lyrics failure is non-fatal, see I1)
    assert r.exit_code == 0, r.output
    assert "letra indisponível" in r.output and "nenhum modelo Whisper" in r.output
    assert count["n"] == 1
    r = runner.invoke(app, ["lyrics", str(audio), "--out", str(out), "--yes", "--model", "medium.en", "--language", "en"])
    assert r.exit_code == 1 and "nenhum modelo Whisper" in r.output   # the lyrics subcommand still fails


def test_lyrics_skips_whisper_when_no_vocal_activity(tmp_path, monkeypatch):
    import json
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"
    calls = []
    monkeypatch.setattr(pipeline, "read_tags", lambda p, **k: Tags(title="T", artist="A"))
    monkeypatch.setattr(pipeline, "separate", lambda a, o, **k: {s: (Path(o) / f"{s}.wav") for s in ["vocals", "guitar", "bass", "drums", "piano", "other"]})
    monkeypatch.setattr(pipeline, "to_wav", lambda s, d, **k: Path(d).write_bytes(b"RIFF") or Path(d))
    monkeypatch.setattr(pipeline, "vocal_activity_for", lambda p, **k: [])
    monkeypatch.setattr(pipeline, "ensure_whisper_model", lambda n, confirm, **k: calls.append("ensure_model") or Path("/m.bin"))
    monkeypatch.setattr(pipeline, "transcribe", lambda w, m, **k: calls.append("transcribe") or [LyricLine(1, 2, "hi", 0.9)])
    monkeypatch.setattr(pipeline, "analyze", lambda stems, **k: Harmony(100, "4/4", "C", 0.0, 2.4, [], []))
    monkeypatch.setattr(pipeline, "transcribe_instrument", lambda w, inst, h, **k: [Note(0.1, 0.4, 60, 0.9, string=1, fret=1)])
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    assert r.exit_code == 0, r.output
    assert "transcribe" not in calls and "ensure_model" not in calls
    assert json.loads((out / "lyrics" / "lyrics.json").read_text()) == []


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
    monkeypatch.setattr(pipeline, "vocal_activity_for", lambda p, **k: [(0.0, 100.0)])
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


def _mock_all(monkeypatch, calls):
    monkeypatch.setattr(pipeline, "read_tags", lambda p, **k: calls.append("tags") or Tags(title="T", artist="A"))
    monkeypatch.setattr(pipeline, "separate", lambda a, o, **k: calls.append("stems") or {s: (Path(o) / f"{s}.wav") for s in ["vocals", "guitar", "bass", "drums", "piano", "other"]})
    monkeypatch.setattr(pipeline, "to_wav", lambda s, d, **k: Path(d).write_bytes(b"RIFF") or Path(d))
    monkeypatch.setattr(pipeline, "vocal_activity_for", lambda p, **k: [(0.0, 100.0)])
    monkeypatch.setattr(pipeline, "ensure_whisper_model", lambda n, confirm, **k: Path("/m.bin"))
    monkeypatch.setattr(pipeline, "transcribe", lambda w, m, **k: calls.append("lyrics") or [LyricLine(1, 2, "hi", 0.9)])
    monkeypatch.setattr(pipeline, "analyze", lambda stems, **k: calls.append("harmony") or Harmony(100, "4/4", "C", 0.0, 2.4, [Chord(1, 0.0, "C", "C", "maj", "C", 0.5)], []))
    monkeypatch.setattr(pipeline, "transcribe_instrument", lambda w, inst, h, **k: calls.append(f"notes:{inst}") or [Note(0.1, 0.4, 60, 0.9, string=1, fret=1)])


def test_run_continues_when_lyrics_fail(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"
    calls = []
    _mock_all(monkeypatch, calls)

    def crash(w, m, **k):
        raise RuntimeError("whisper-cli failed: segfault")
    monkeypatch.setattr(pipeline, "transcribe", crash)
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    assert r.exit_code == 0, r.output
    assert "letra indisponível: whisper-cli failed" in r.output
    assert (out / "render" / "cifra.html").exists()
    assert not (out / "lyrics" / "lyrics.json").exists() and not (out / "lyrics" / ".done").exists()
    assert "Letra não transcrita" in (out / "render" / "cifra.html").read_text(encoding="utf-8")


def test_run_with_changed_audio_reruns_stages(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"
    calls = []
    _mock_all(monkeypatch, calls)
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    assert r.exit_code == 0, r.output
    calls.clear()
    audio.write_bytes(b"y")
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    assert r.exit_code == 0, r.output
    assert "áudio mudou, cache invalidado" in r.output
    assert calls == ["tags", "stems", "lyrics", "harmony", "notes:guitar"]


def test_stems_force_invalidates_downstream(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"
    calls = []
    _mock_all(monkeypatch, calls)
    runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    calls.clear()
    r = runner.invoke(app, ["stems", str(audio), "--out", str(out), "--force"])
    assert r.exit_code == 0 and calls == ["stems"]
    calls.clear()
    runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    assert calls == ["lyrics", "harmony", "notes:guitar"]


def test_notes_marks_done_per_instrument(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"
    calls = []
    _mock_all(monkeypatch, calls)
    runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    calls.clear()
    r = runner.invoke(app, ["notes", str(audio), "--out", str(out), "--instruments", "guitar,bass"])
    assert r.exit_code == 0 and calls == ["notes:bass"]
    assert (out / "notes" / ".done.guitar").exists() and (out / "notes" / ".done.bass").exists()


def test_subcommands_accept_device(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"
    seen = []
    _mock_all(monkeypatch, [])
    monkeypatch.setattr(pipeline, "separate", lambda a, o, **k: seen.append(k["device"]) or {})
    r = runner.invoke(app, ["harmony", str(audio), "--out", str(out), "--device", "cpu"])
    assert r.exit_code == 0, r.output
    assert seen == ["cpu"]
    for cmd in ("lyrics", "notes"):
        assert "--device" in runner.invoke(app, [cmd, "--help"]).output


def test_missing_non_tool_file_has_no_brew_hint(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    r = runner.invoke(app, ["render", str(audio), "--out", str(tmp_path / "out")])   # no harmony.json yet
    assert r.exit_code == 1
    assert "brew install" not in r.output and "harmony.json" in r.output
