import numpy as np
from music_transcribe.harmony import tempo as tempo_module
from music_transcribe.harmony.tempo import estimate_tempo
from conftest import click_track, SR


def test_shuffle_52_is_not_reported_as_156():
    np.random.seed(0)
    y = click_track(52, bars=8, sr=SR, subdiv=3)           # 12/8 feel at 52 BPM
    bar = 4 * 60 / 52
    changes = [0.0 + k * bar for k in range(8)]
    bpm, meter = estimate_tempo(y, SR, changes)
    assert abs(bpm - 52) < 3
    assert meter == "12/8"


def test_straight_120_is_4_4():
    np.random.seed(0)
    y = click_track(120, bars=8, sr=SR, subdiv=2)
    bar = 4 * 60 / 120
    changes = [k * bar for k in range(8)]
    bpm, meter = estimate_tempo(y, SR, changes)
    assert abs(bpm - 120) < 3
    assert meter == "4/4"


def test_empty_candidates_fallback(monkeypatch):
    monkeypatch.setattr(tempo_module, "_tempo_candidates", lambda *a, **k: [])
    assert estimate_tempo(np.zeros(SR), SR, []) == (120.0, "4/4")


def test_few_root_changes_still_octave_corrects():
    np.random.seed(0)
    y = click_track(52, bars=4, sr=SR, subdiv=3)
    bpm, meter = estimate_tempo(y, SR, [0.0])
    assert abs(bpm - 52) < 3
