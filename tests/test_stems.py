from pathlib import Path
from music_transcribe.stems import separate, STEM_NAMES


def test_separate_invokes_demucs_and_moves_stems(tmp_path):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "stems"
    def runner(cmd, **kw):
        assert cmd[1:3] == ["-m", "demucs"]
        assert "htdemucs_6s" in cmd
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
