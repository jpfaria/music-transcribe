from pathlib import Path
from music_transcribe.models import ensure_whisper_model, WHISPER_MODELS


def test_existing_model_returns_without_confirm(tmp_path):
    p = tmp_path / "ggml-large-v3.bin"; p.write_bytes(b"x" * 10)
    called = []
    out = ensure_whisper_model("large-v3", confirm=lambda n, mb: called.append(1) or False, cache_dir=tmp_path)
    assert out == p and not called


def test_declined_download_returns_none(tmp_path):
    out = ensure_whisper_model("large-v3", confirm=lambda n, mb: False, cache_dir=tmp_path, downloader=lambda u, d: None)
    assert out is None


def test_accepted_download_calls_downloader_with_url(tmp_path):
    got = {}
    def dl(url, dst): got["url"] = url; Path(dst).write_bytes(b"model")
    out = ensure_whisper_model("medium.en", confirm=lambda n, mb: True, cache_dir=tmp_path, downloader=dl)
    assert out and out.exists()
    assert got["url"] == WHISPER_MODELS["medium.en"][0]


def test_medium_multilingual_model_is_registered():
    assert WHISPER_MODELS["medium"][0].endswith("ggml-medium.bin")


def test_unknown_model_lists_known_names(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="large-v3"):
        ensure_whisper_model("huge", confirm=lambda n, mb: True, cache_dir=tmp_path)


def test_reuses_whisper_cpp_cache_without_confirm(tmp_path):
    shared = tmp_path / "whisper"; shared.mkdir()
    p = shared / "ggml-large-v3.bin"; p.write_bytes(b"x" * 10)
    out = ensure_whisper_model("large-v3", confirm=lambda n, mb: (_ for _ in ()).throw(AssertionError("asked")),
                               cache_dir=tmp_path / "cache", reuse_dir=shared)
    assert out == p


def test_curl_failure_becomes_runtime_error(tmp_path, monkeypatch):
    import subprocess
    import pytest
    from music_transcribe import models

    def fail(cmd, **k):
        raise subprocess.CalledProcessError(22, cmd)
    monkeypatch.setattr(models.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="download falhou"):
        models._curl("https://example.invalid/x.bin", tmp_path / "x.bin")


def test_unknown_model_via_cli_is_a_clean_error(tmp_path, monkeypatch):
    from pathlib import Path
    from typer.testing import CliRunner
    from music_transcribe.cli import app
    from music_transcribe import pipeline
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "separate", lambda a, o, **k: {})
    monkeypatch.setattr(pipeline, "to_wav", lambda s, d, **k: Path(d).write_bytes(b"RIFF") or Path(d))
    monkeypatch.setattr(pipeline, "vocal_activity_for", lambda p, **k: [(0.0, 100.0)])
    r = CliRunner().invoke(app, ["lyrics", str(audio), "--out", str(tmp_path / "o"), "--model", "huge"])
    assert r.exit_code == 1 and "modelo Whisper desconhecido" in r.output
