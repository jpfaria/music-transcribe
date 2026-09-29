from pathlib import Path
from music_transcribe.stems import separate, STEM_NAMES


def test_separate_invokes_demucs_and_moves_stems(tmp_path):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "stems"
    def runner(cmd, **kw):
        assert cmd[1:3] == ["-m", "demucs"]
        assert "htdemucs_6s" in cmd
        assert cmd[cmd.index("--shifts") + 1] == "0"      # deterministic separation
        # emulate demucs output layout: <o>/htdemucs_6s/<track>/<stem>.wav
        d = Path(cmd[cmd.index("-o") + 1]) / "htdemucs_6s" / "song"
        d.mkdir(parents=True)
        for s in STEM_NAMES:
            (d / f"{s}.wav").write_bytes(b"RIFF")
        class R: returncode = 0; stderr = ""
        return R()
    stems = separate(audio, out, runner=runner)
    assert set(stems) == set(STEM_NAMES)
    assert all(p.exists() and p.parent == out for p in stems.values())


def test_separate_missing_stem_raises(tmp_path):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    def runner(cmd, **kw):
        d = Path(cmd[cmd.index("-o") + 1]) / "htdemucs_6s" / "song"; d.mkdir(parents=True)
        (d / "vocals.wav").write_bytes(b"RIFF")
        class R: returncode = 0; stderr = ""
        return R()
    import pytest
    with pytest.raises(RuntimeError, match="guitar"):
        separate(audio, tmp_path / "o", runner=runner)


def _ok_runner(calls, fail_first=False):
    def runner(cmd, **kw):
        calls.append((cmd, kw))
        class R: returncode = 0; stderr = ""
        if fail_first and len(calls) == 1:
            R.returncode = 1; R.stderr = "MPS backend out of memory"
            return R()
        d = Path(cmd[cmd.index("-o") + 1]) / "htdemucs_6s" / "song"
        d.mkdir(parents=True)
        for s in STEM_NAMES:
            (d / f"{s}.wav").write_bytes(b"RIFF")
        return R()
    return runner


def test_separate_sets_torch_home_under_cache_and_confirms_missing_weights(tmp_path, monkeypatch):
    from music_transcribe.models import default_cache_dir
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    calls, asked = [], []
    separate(audio, tmp_path / "o", runner=_ok_runner(calls), confirm=lambda n, mb: asked.append((n, mb)) or True)
    env = calls[0][1]["env"]
    assert Path(env["TORCH_HOME"]) == default_cache_dir() / "torch"
    assert asked == [("htdemucs_6s", 55)]


def test_separate_declined_weights_aborts(tmp_path):
    import pytest
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    calls = []
    with pytest.raises(RuntimeError, match="demucs"):
        separate(audio, tmp_path / "o", runner=_ok_runner(calls), confirm=lambda n, mb: False)
    assert calls == []


def test_separate_skips_confirm_when_checkpoint_cached(tmp_path):
    from music_transcribe.stems import torch_home
    ck = torch_home() / "hub" / "checkpoints"; ck.mkdir(parents=True)
    (ck / "5c90dfd2-34c22ccb.th").write_bytes(b"w")
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    separate(audio, tmp_path / "o", runner=_ok_runner([]), confirm=lambda n, mb: (_ for _ in ()).throw(AssertionError("asked")))


def test_ensure_weights_links_existing_torch_hub_checkpoint(tmp_path):
    from music_transcribe.stems import ensure_demucs_weights
    sysd = tmp_path / "sys" / "hub" / "checkpoints"; sysd.mkdir(parents=True)
    (sysd / "5c90dfd2-34c22ccb.th").write_bytes(b"w")
    home = tmp_path / "home"
    ensure_demucs_weights(home, confirm=lambda n, mb: False, reuse_dirs=[sysd])
    linked = home / "hub" / "checkpoints" / "5c90dfd2-34c22ccb.th"
    assert linked.is_symlink() and linked.read_bytes() == b"w"


def test_separate_retries_on_cpu_after_device_failure(tmp_path):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    calls = []
    stems = separate(audio, tmp_path / "o", runner=_ok_runner(calls, fail_first=True), device="mps")
    assert len(calls) == 2
    first, second = calls[0][0], calls[1][0]
    assert first[first.index("-d") + 1] == "mps" and second[second.index("-d") + 1] == "cpu"
    assert set(stems) == set(STEM_NAMES)


def test_separate_no_retry_when_already_cpu(tmp_path):
    import pytest
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    calls = []
    def runner(cmd, **kw):
        calls.append(cmd)
        class R: returncode = 1; stderr = "boom"
        return R()
    with pytest.raises(RuntimeError, match="demucs failed"):
        separate(audio, tmp_path / "o", runner=runner, device="cpu")
    assert len(calls) == 1
