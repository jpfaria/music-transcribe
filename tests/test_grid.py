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


def test_refine_bar_len_from_bass_changes():
    import numpy as np
    from music_transcribe.harmony.grid import refine_bar_len
    jit = np.random.default_rng(0).uniform(-0.1, 0.1, 12)
    t, changes = 0.0, []
    for i, d in enumerate(jit):
        changes.append(t + d)
        t += 9.3 if i == 5 else 4.64          # one two-bar gap (k=2)
    assert abs(refine_bar_len(changes, 4.46) - 4.64) <= 0.05


def test_refine_bar_len_needs_four_usable_intervals():
    from music_transcribe.harmony.grid import refine_bar_len
    assert refine_bar_len([0.0, 4.64, 9.28, 13.92], 4.46) == 4.46          # only 3 intervals
    assert refine_bar_len([0.0, 1.5, 3.0, 4.5, 6.0, 7.5], 4.46) == 4.46    # r≈0.34: not a whole bar
