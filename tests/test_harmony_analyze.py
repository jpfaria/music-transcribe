import numpy as np
import pytest
from music_transcribe.harmony import analyze
from music_transcribe.schema import Harmony
from conftest import sine_note, midi_to_hz, write_wav, SR


def test_analyze_handles_ragged_stem_lengths(tmp_path):
    # durations deliberately unequal to exercise the ragged-stem padding fix; long enough
    # (>~3s) that librosa's default chroma_cqt octave count doesn't warn on short input.
    bass = sine_note(midi_to_hz(34), 3.0, SR, harmonics=2)  # Bb1
    guitar = sine_note(midi_to_hz(58), 3.1, SR, harmonics=3)
    other = sine_note(midi_to_hz(62), 2.9, SR, harmonics=3)
    drums = (0.2 * np.random.default_rng(0).standard_normal(int(3.4 * SR))).astype(np.float32)

    stems = {
        "bass": write_wav(tmp_path / "bass.wav", bass),
        "guitar": write_wav(tmp_path / "guitar.wav", guitar),
        "other": write_wav(tmp_path / "other.wav", other),
        "drums": write_wav(tmp_path / "drums.wav", drums),
    }

    h = analyze(stems, sr=SR)
    assert isinstance(h, Harmony)
    assert h.bar_len > 0
    assert isinstance(h.chords, list)


def test_analyze_requires_harmonic_stem():
    with pytest.raises(RuntimeError, match="harmonic"):
        analyze({"drums": "does/not/exist.wav"})
