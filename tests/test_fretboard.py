from music_transcribe.notes.fretboard import assign_positions, preferred_region, TUNINGS
from music_transcribe.schema import Note


def test_positions_stay_close():
    notes = [Note(i * 0.5, i * 0.5 + 0.4, p, 0.8) for i, p in enumerate([58, 61, 63, 65, 68])]
    assign_positions(notes, TUNINGS["guitar"], preferred=(6, 10))
    frets = [n.fret for n in notes]
    assert all(f is not None for f in frets)
    assert max(frets) - min(frets) <= 5


def test_unplayable_pitch_left_none():
    n = [Note(0, 1, 30, 0.8)]
    assign_positions(n, TUNINGS["guitar"])
    assert n[0].fret is None


def test_bass_tuning():
    n = [Note(0, 1, 46, 0.8)]  # Bb2
    assign_positions(n, TUNINGS["bass"])
    assert TUNINGS["bass"][n[0].string] + n[0].fret == 46


def test_preferred_region_for_key():
    assert preferred_region("Bbm") == (4, 10)   # Bb on 6th string = fret 6
    assert preferred_region("E") == (0, 4)
