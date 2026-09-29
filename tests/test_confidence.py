from music_transcribe.notes.confidence import score, mark_bleed_runs
from music_transcribe.schema import Note


def test_score_levels():
    assert score(Note(0, 1, 60, 0.9), voiced_frac=0.9, repeated_low_run=False) == ("high", "")
    assert score(Note(0, 1, 60, 0.5), voiced_frac=0.9, repeated_low_run=False) == ("medium", "weak")
    assert score(Note(0, 1, 60, 0.9), voiced_frac=0.3, repeated_low_run=False) == ("medium", "unstable")
    assert score(Note(0, 1, 46, 0.9), voiced_frac=0.9, repeated_low_run=True) == ("low", "bleed")


def test_mark_bleed_runs_finds_triplet_drone():
    notes = [Note(i * 0.3, i * 0.3 + 0.25, 46, 0.6) for i in range(8)] + [Note(3.0, 3.4, 70, 0.8)]
    idx = mark_bleed_runs(notes)
    assert idx == set(range(8))
