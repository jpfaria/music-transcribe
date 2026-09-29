"""Synthesize a 30 s 'song' in Bbm at 60 BPM 4/4: bass roots Bb-Db-Eb-Bb per bar, chord pad, a lead line with a bend."""
import numpy as np
from conftest import sine_note, midi_to_hz, SR


def make(path):
    bar = 4.0
    roots = [34, 37, 39, 34]                       # Bb1 Db2 Eb2 Bb1
    tri = {34: [58, 61, 65], 37: [61, 65, 68], 39: [63, 66, 70]}
    y = np.zeros(int(8 * bar * SR) + SR, dtype=np.float32)
    lead = [70, 73, 75, 77, 75, 73, 70, 68]
    for b in range(8):
        r = roots[b % 4]; t0 = int(b * bar * SR)
        for k in range(4):
            seg = sine_note(midi_to_hz(r), 0.9, SR, amp=0.4, harmonics=2)
            y[t0 + k * SR: t0 + k * SR + len(seg)] += seg
        for p in tri[r]:
            seg = sine_note(midi_to_hz(p), bar, SR, amp=0.12)
            y[t0: t0 + len(seg)] += seg
        seg = sine_note(midi_to_hz(lead[b]), 1.5, SR, amp=0.35)
        y[t0: t0 + len(seg)] += seg
    import soundfile as sf
    sf.write(str(path), y / (np.abs(y).max() + 1e-6) * 0.9, SR)
    return path
