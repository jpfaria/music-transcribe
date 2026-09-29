from music_transcribe.harmony.grid import fit_grid


def test_fit_grid_recovers_offset():
    bar = 4.64
    changes = [0.75 + k * bar + d for k, d in zip(range(6), [0.0, 0.1, -0.1, 0.05, 0.0, -0.05])]
    assert abs(fit_grid(changes, bar) - 0.75) < 0.08


def test_fit_grid_first_bar_at_or_before_first_change():
    changes = [0.3 + k * 2.0 for k in range(4)]
    bar0 = fit_grid(changes, 2.0)
    assert -2.0 < bar0 <= 0.3
