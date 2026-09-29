import numpy as np
from music_transcribe.notes.detect import raw_notes, f0_contour
from music_transcribe.schema import Note
from conftest import sine_note, midi_to_hz, SR


def test_raw_notes_monophonic_reduction(tmp_path):
    fake = [(0.0, 0.5, 58, 0.9), (0.02, 0.5, 46, 0.3), (1.0, 1.5, 61, 0.8), (1.0, 1.2, 65, 0.2)]
    notes = raw_notes(tmp_path / "x.wav", predictor=lambda p: fake, min_amp=0.25)
    assert [n.pitch for n in notes] == [58, 61]
    # 46 was suppressed under 58 → the kept note is flagged as a probable chord
    assert (notes[0].confidence, notes[0].reason) == ("medium", "chord")
    # 65 fell under min_amp, nothing was suppressed → 61 untouched
    assert (notes[1].confidence, notes[1].reason) == ("high", "")


def test_raw_notes_piano_keeps_simultaneous_notes(tmp_path):
    fake = [(0.0, 0.5, 60, 0.9), (0.01, 0.5, 64, 0.7), (0.02, 0.5, 67, 0.6)]
    notes = raw_notes(tmp_path / "x.wav", predictor=lambda p: fake, instrument="piano")
    assert [n.pitch for n in notes] == [60, 64, 67]
    assert all(n.reason == "" for n in notes)


def test_raw_notes_default_min_amp_keeps_weak_notes():
    from music_transcribe.notes.confidence import score
    notes = raw_notes("x.wav", predictor=lambda p: [(0.0, 0.5, 60, 0.32)])
    assert len(notes) == 1
    assert score(notes[0], 1.0, False) == ("low", "weak")      # the weak branch is reachable


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
