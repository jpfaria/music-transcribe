import numpy as np
from conftest import sine_note, midi_to_hz, write_wav
from music_transcribe.notes import transcribe_instrument
from music_transcribe.schema import Harmony


def test_transcribe_instrument_assigns_positions(tmp_path):
    sr = 22050
    p1, p2 = 58, 61
    y = np.concatenate([
        sine_note(midi_to_hz(p1), 0.4, sr=sr),
        sine_note(midi_to_hz(p2), 0.4, sr=sr),
        np.zeros(int(0.2 * sr), dtype=np.float32),
    ])
    wav = write_wav(tmp_path / "notes.wav", y, sr=sr)

    def fake_predictor(path):
        return [(0.0, 0.4, p1, 0.8), (0.45, 0.85, p2, 0.8)]

    harmony = Harmony(bpm=120.0, meter="4/4", key="Bbm", bar0=0.0, bar_len=2.0)
    notes = transcribe_instrument(wav, "guitar", harmony, predictor=fake_predictor)

    assert len(notes) == 2
    assert all(n.fret is not None for n in notes)
    assert all(n.string is not None for n in notes)
