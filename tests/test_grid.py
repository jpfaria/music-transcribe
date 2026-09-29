from music_transcribe.harmony.grid import fit_grid


def test_fit_grid_recovers_offset_as_pickup_bar():
    bar = 4.64
    changes = [0.75 + k * bar + d for k, d in zip(range(6), [0.0, 0.1, -0.1, 0.05, 0.0, -0.05])]
    # phase 0.75 > 5% of a bar → bar 1 is a pickup that starts before t=0
    assert abs(fit_grid(changes, bar) - (0.75 - bar)) < 0.08


def test_fit_grid_pickup_bar_covers_t0():
    changes = [0.3 + k * 2.0 for k in range(4)]
    bar0 = fit_grid(changes, 2.0)
    assert abs(bar0 - (0.3 - 2.0)) < 0.01


def test_fit_grid_song_starts_on_downbeat():
    changes = [0.0 + k * 4.0 for k in range(6)]
    assert abs(fit_grid(changes, 4.0)) < 0.3


def test_fit_grid_late_bass_entry_keeps_intro():
    changes = [16.0 + k * 4.0 for k in range(6)]
    bar0 = fit_grid(changes, 4.0)
    assert abs(bar0) < 0.05          # not 16 (or 4): bar 1 still covers t=0


def test_fit_grid_phase_just_before_bar_wraps_negative():
    changes = [3.9 + k * 4.0 for k in range(6)]
    assert abs(fit_grid(changes, 4.0) - (-0.1)) < 0.02


def test_fit_grid_jittered_downbeat_within_tolerance():
    changes = [k * 4.0 + d for k, d in zip(range(6), [0.05, -0.05, 0.1, -0.1, 0.0, 0.05])]
    bar0 = fit_grid(changes, 4.0)
    assert -4.0 < bar0 <= 0.2 and abs(bar0) < 0.1
