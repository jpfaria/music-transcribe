import numpy as np
from music_transcribe.notes.articulation import classify_contour, detect_legato
from music_transcribe.schema import Note

HOP = 256 / 22050


def ramp(n, a, b):
    return np.linspace(a, b, n)


def test_flat_is_plain():
    assert classify_contour(np.zeros(60) + np.random.randn(60) * 5, HOP) == ""


def test_full_bend():
    c = np.concatenate([np.zeros(20), ramp(20, 0, 200), np.full(30, 200)])
    assert classify_contour(c, HOP) == "bend:2"


def test_half_bend():
    c = np.concatenate([np.zeros(20), ramp(20, 0, 100), np.full(30, 100)])
    assert classify_contour(c, HOP) == "bend:1"


def test_bend_release():
    c = np.concatenate([np.zeros(10), ramp(20, 0, 200), np.full(10, 200), ramp(20, 200, 0), np.zeros(10)])
    assert classify_contour(c, HOP) == "bend-release"


def test_vibrato():
    t = np.arange(100) * HOP
    c = 40 * np.sin(2 * np.pi * 6 * t)
    assert classify_contour(c, HOP) == "vibrato"


def test_nan_heavy_is_plain():
    c = np.full(60, np.nan); c[:5] = 0
    assert classify_contour(c, HOP) == ""


def test_slide_when_gap_is_voiced():
    p, c = Note(0, 0.5, 58, 0.8), Note(0.52, 1.0, 61, 0.8)
    assert detect_legato(p, c, gap_voiced_frac=0.9, onset_ratio=0.9) == "slide:61"


def test_hammer_and_pull_on_weak_onset():
    p, c = Note(0, 0.5, 58, 0.8), Note(0.5, 1.0, 61, 0.8)
    assert detect_legato(p, c, gap_voiced_frac=0.0, onset_ratio=0.3) == "hammer"
    p2, c2 = Note(0, 0.5, 61, 0.8), Note(0.5, 1.0, 58, 0.8)
    assert detect_legato(p2, c2, gap_voiced_frac=0.0, onset_ratio=0.3) == "pull"


def test_normal_attack_is_plain():
    p, c = Note(0, 0.5, 58, 0.8), Note(0.6, 1.0, 61, 0.8)
    assert detect_legato(p, c, gap_voiced_frac=0.1, onset_ratio=1.0) == ""
