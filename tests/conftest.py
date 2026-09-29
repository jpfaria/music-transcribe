import numpy as np
import pytest
import soundfile as sf

SR = 22050


def sine_note(freq: float, dur: float, sr: int = SR, amp: float = 0.5, harmonics: int = 4) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    y = sum((amp / (k + 1)) * np.sin(2 * np.pi * freq * (k + 1) * t) for k in range(harmonics))
    env = np.minimum(1.0, t / 0.01) * np.exp(-t * 1.5)
    return (y * env).astype(np.float32)


def midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def click_track(bpm: float, bars: int, sr: int = SR, subdiv: int = 3, beats_per_bar: int = 4) -> np.ndarray:
    beat = 60.0 / bpm
    total = int(bars * beats_per_bar * beat * sr) + sr
    y = np.zeros(total, dtype=np.float32)
    n = 0
    while True:
        for s in range(subdiv):
            t = (n + s / subdiv) * beat
            i = int(t * sr)
            if i >= total - 200:
                return y
            amp = 1.0 if s == 0 else 0.4
            y[i:i + 200] += amp * np.hanning(200) * np.sign(np.random.randn(200))
        n += 1


def write_wav(path, y: np.ndarray, sr: int = SR):
    sf.write(str(path), y, sr)
    return path


@pytest.fixture
def sr():
    return SR
