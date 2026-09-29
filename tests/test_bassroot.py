import numpy as np
from music_transcribe.harmony.bassroot import bass_roots, root_changes
from conftest import sine_note, midi_to_hz, SR


def test_bass_roots_follow_sequence():
    # Bb1 (34), Db2 (37), Eb2 (39), Bb1 — 2 s each
    y = np.concatenate([sine_note(midi_to_hz(m), 2.0, SR, harmonics=2) for m in [34, 37, 39, 34]])
    runs = bass_roots(y, SR)
    pcs = [r.pc for r in runs if r.dur >= 1.0]
    assert pcs == [10, 1, 3, 10]
    ch = root_changes(runs)
    assert len(ch) == 4 and abs(ch[1] - 2.0) < 0.3
