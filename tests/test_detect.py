import numpy as np
from music_transcribe.notes.detect import raw_notes, f0_contour
from music_transcribe.schema import Note
from conftest import sine_note, midi_to_hz, SR


def test_raw_notes_monophonic_reduction(tmp_path):
    fake = [(0.0, 0.5, 58, 0.9), (0.02, 0.5, 46, 0.3), (1.0, 1.5, 61, 0.8), (1.0, 1.2, 65, 0.2)]
    notes = raw_notes(tmp_path / "x.wav", predictor=lambda p: fake, min_amp=0.25)
    assert [n.pitch for n in notes] == [58, 61]


def test_f0_contour_flat_note_is_near_zero_cents():
    y = sine_note(midi_to_hz(58), 1.0, SR)
    cents, voiced = f0_contour(y, SR, Note(0.0, 1.0, 58, 0.9))
    assert voiced > 0.8
    assert abs(np.nanmedian(cents)) < 25


def test_f0_contour_bend_rises():
    t = np.arange(int(1.0 * SR)) / SR
    f = midi_to_hz(58) * 2 ** (np.clip((t - 0.3) / 0.4, 0, 1) * 2 / 12)   # bend up 2 semitones
    y = (0.5 * np.sin(2 * np.pi * np.cumsum(f) / SR)).astype(np.float32)
    cents, voiced = f0_contour(y, SR, Note(0.0, 1.0, 58, 0.9))
    assert np.nanmedian(cents[-len(cents)//4:]) - np.nanmedian(cents[:len(cents)//4]) > 150
