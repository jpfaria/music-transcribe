import numpy as np
from conftest import sine_note, midi_to_hz, write_wav
from music_transcribe.notes import transcribe_instrument
from music_transcribe.schema import Harmony

SR = 22050


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


def _note_with_quiet_middle(f1, f2, dur, sr, head_dur, tail_dur, amp=0.5, harmonics=4):
    """A single note whose pitch trace is only reliably voiced near its edges:
    a clear flat tone at f1, then near-silence, then a clear flat tone at f2.
    Used to give the note a strong, sign-consistent contour "drift" toward f2
    while staying under classify_contour's voiced-coverage gate (so it is not
    itself classified as a same-note bend)."""
    n = int(dur * sr)
    n_head, n_tail = int(head_dur * sr), int(tail_dur * sr)
    n_mid = n - n_head - n_tail
    t_head = np.arange(n_head) / sr
    y_head = sum((amp / (k + 1)) * np.sin(2 * np.pi * f1 * (k + 1) * t_head) for k in range(harmonics))
    y_head = y_head * np.minimum(1.0, t_head / 0.01)
    y_mid = np.zeros(n_mid, dtype=np.float32)
    t_tail = np.arange(n_tail) / sr
    y_tail = sum((amp / (k + 1)) * np.sin(2 * np.pi * f2 * (k + 1) * t_tail) for k in range(harmonics))
    y_tail = y_tail * np.minimum(1.0, t_tail / 0.01) * np.exp(-t_tail * 1.0)
    return np.concatenate([y_head, y_mid, y_tail]).astype(np.float32)


def test_touching_pair_weak_onset_is_hammer(tmp_path):
    # A run of ordinary picked decoy notes establishes what a "normal" onset
    # looks like, so a touching join with no new pick attack reads as weak
    # (onset_ratio < 0.5) relative to that baseline -- the hammer-on case.
    p1, p2 = 58, 61
    decoy_pitches = [50 + (i % 6) for i in range(24)]
    decoys = [sine_note(midi_to_hz(p), 0.06, sr=SR, amp=0.85) for p in decoy_pitches]
    half = len(decoys) // 2
    pre, post = np.concatenate(decoys[:half]), np.concatenate(decoys[half:])
    # One continuous, unbroken tone under the touching pair: no physical
    # discontinuity at the join, matching a real hammer-on's single pick.
    pair = sine_note(midi_to_hz(p1), 0.8, sr=SR, amp=0.5)
    y = np.concatenate([pre, pair, post])
    p1_start = len(pre) / SR
    wav = write_wav(tmp_path / "hammer.wav", y, sr=SR)

    def fake_predictor(path):
        return [(p1_start, p1_start + 0.4, p1, 0.8), (p1_start + 0.4, p1_start + 0.8, p2, 0.8)]

    harmony = Harmony(bpm=120.0, meter="4/4", key="Bbm", bar0=0.0, bar_len=2.0)
    notes = transcribe_instrument(wav, "guitar", harmony, predictor=fake_predictor)

    assert notes[1].articulation == "hammer"


def test_touching_pair_contour_drift_is_slide(tmp_path):
    p1, p2 = 58, 61
    y1 = _note_with_quiet_middle(midi_to_hz(p1), midi_to_hz(p2), 0.8, SR, head_dur=0.08, tail_dur=0.12)
    y2 = sine_note(midi_to_hz(p2), 0.3, sr=SR, amp=0.5)
    y = np.concatenate([y1, y2])
    wav = write_wav(tmp_path / "slide.wav", y, sr=SR)

    def fake_predictor(path):
        return [(0.0, 0.8, p1, 0.8), (0.8, 1.1, p2, 0.8)]

    harmony = Harmony(bpm=120.0, meter="4/4", key="Bbm", bar0=0.0, bar_len=2.0)
    notes = transcribe_instrument(wav, "guitar", harmony, predictor=fake_predictor)

    assert notes[0].articulation == "slide:61"


def test_transcribe_instrument_keeps_chord_flag(tmp_path):
    y = sine_note(midi_to_hz(58), 0.6, sr=SR)
    wav = write_wav(tmp_path / "c.wav", y, sr=SR)
    fake = lambda p: [(0.0, 0.5, 58, 0.9), (0.01, 0.5, 62, 0.5)]
    notes = transcribe_instrument(wav, "guitar", Harmony(120.0, "4/4", "Bbm", 0.0, 2.0), predictor=fake)
    assert [n.pitch for n in notes] == [58]
    assert (notes[0].confidence, notes[0].reason) == ("medium", "chord")


def test_transcribe_instrument_piano_passes_instrument(tmp_path):
    y = sine_note(midi_to_hz(60), 0.6, sr=SR)
    wav = write_wav(tmp_path / "p.wav", y, sr=SR)
    fake = lambda p: [(0.0, 0.5, 60, 0.9), (0.01, 0.5, 64, 0.8)]
    notes = transcribe_instrument(wav, "piano", Harmony(120.0, "4/4", "C", 0.0, 2.0), predictor=fake)
    assert sorted(n.pitch for n in notes) == [60, 64]
