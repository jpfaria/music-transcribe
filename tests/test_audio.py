import numpy as np
from music_transcribe.audio import load_mono, to_wav, duration
from conftest import sine_note, write_wav


def test_load_mono_resamples_stereo_48k(tmp_path):
    y = sine_note(440, 1.0, sr=48000)
    stereo = np.stack([y, y], axis=1)
    p = write_wav(tmp_path / "s.wav", stereo, 48000)
    m, sr = load_mono(p, sr=22050)
    assert sr == 22050 and m.ndim == 1
    assert abs(len(m) - 22050) < 50


def test_to_wav_uses_runner(tmp_path):
    calls = []
    def runner(cmd, **kw):
        calls.append(cmd); (tmp_path / "o.wav").write_bytes(b"RIFF")
        class R: returncode = 0
        return R()
    out = to_wav(tmp_path / "in.flac", tmp_path / "o.wav", sr=16000, mono=True, runner=runner)
    assert out.exists()
    cmd = calls[0]
    assert cmd[0] == "ffmpeg" and "-ar" in cmd and "16000" in cmd and "-ac" in cmd


def test_duration(tmp_path):
    p = write_wav(tmp_path / "d.wav", sine_note(220, 2.0), 22050)
    assert abs(duration(p) - 2.0) < 0.01
