from music_transcribe.harmony.grid import fit_grid


def test_fit_grid_recovers_offset():
    bar = 4.64
    changes = [0.75 + k * bar + d for k, d in zip(range(6), [0.0, 0.1, -0.1, 0.05, 0.0, -0.05])]
    assert abs(fit_grid(changes, bar) - 0.75) < 0.08
