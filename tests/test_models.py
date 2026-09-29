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
