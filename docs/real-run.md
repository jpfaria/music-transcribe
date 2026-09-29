# Real-song smoke run (2026-09-29)

Command (deviation: `--model small.en`; `ggml-large-v3.bin` exists in neither `~/.cache/whisper/` nor
`~/.cache/music-transcribe/`, so large-v3 would have been a new 3 GB download):

```
uv run music-transcribe run ~/Music/my_music.flac --yes --device mps --language en --model small.en
```

Input: `my_music.flac`, 3:36 (216.4 s), tags "Midnight on My Mind" / Chicago Blues Radio, no embedded lyrics.
Output: `~/Music/my_music.transcribe/` (input.sha1 `fb10a6cfad26`). Code at commit 0892840.

## Time per stage (wall clock, first run, all models already cached)
| tags | stems (demucs, MPS) | lyrics (whisper small.en) | harmony | notes (guitar, bass, piano) | render | total |
|---|---|---|---|---|---|---|
| <1 s | 39 s | 8 s | 18 s | 50 s | 1 s | 116 s |

MPS worked: demucs did not fall back to cpu (no "tentando de novo em cpu" line). The demucs checkpoint
was reused from `~/.cache/torch/hub` through a symlink in `~/.cache/music-transcribe/torch/hub/checkpoints/`,
with no download and no prompt.

## Results vs the earlier manual analysis
| | pipeline | manual (expected) |
|---|---|---|
| tempo / meter | 53.8 BPM, 12/8 | ~52 BPM, 12/8 |
| key | Bbm | Bbm |
| bar_len / bar0 | 4.461 s / −0.51 s | n/a |
| loop | `[]` (none detected) | Bbm \| Db \| Ebm \| Bbm(F) |
| chords | 49 bars, 43 with confidence < 0.3; first bars Bbm Db Ebsus2 Bbm Bbm Db Dbsus2 Bbsus4 … | |
| lyrics | 9 segments, 0 marked `?` (all ≥ 0.6) | 25 lines |
| solo scale | Bb minor pentatonic, box on fret 6 | |

Lyrics: whisper small.en returns long segments that each hold 2–4 sung lines (e.g. "I got a midnight on my
mind Chasing ghosts I'll never find"), so 9 segments cover about 25 sung lines. The first starts at 29.5 s,
and there is a gap from 101 s to 166 s (probably the solo).

## Notes per instrument (confidence split, reasons)
| instrument | notes | high | medium | low | reasons |
|---|---|---|---|---|---|
| guitar | 892 | 165 | 458 | 269 | weak 303, unstable 249, chord 175 |
| bass | 379 | 99 | 192 | 88 | chord 116, weak 101, unstable 52, bleed 11 |
| piano | 905 | 10 | 95 | 800 | unstable 725, weak 170 |

## Open points for the controller (nothing was tuned)
- No loop was detected, and 43 of 49 chords fall under 0.3 confidence, while the first bars do follow the
  expected Bbm → Db → Eb(m) motion. `Ebsus2`/`Dbsus2`/`Bbsus4` suggest the upper stems' extensions win
  over the triad.
- On bass, 116 of 379 notes carry `chord`: basic-pitch's octave/overtone doubles within 60 ms count as
  suppressed chord tones.
- Piano is 88% `low`/`unstable`: pyin on a polyphonic stem rarely finds a single voiced f0.
- Not run with large-v3 (see above). Lyric line counts will differ with it.

## Re-run after `refine_bar_len` + `--shifts 0` (2026-09-29, `--force`, code = a36ed30 + refine_bar_len)

Command: `uv run music-transcribe run ~/Music/my_music.flac --yes --device mps --language en --model small.en --force`.
Times: tags <1 s, stems 26 s (MPS, no fallback; `--shifts 0` is faster than the first run's 39 s), lyrics 5 s,
harmony 7 s, notes 29 s, render <1 s; 67 s total.

| | first run | re-run |
|---|---|---|
| bpm / bar_len | 53.8 / 4.461 s (tempogram) | 53.3 / 4.500 s (refined) |
| bar0 | −0.51 | −2.186 |
| loop | `[]` | `[]` |
| chords < 0.3 | 43 of 49 | 47 of 49 |
| lyrics | 9 segments, 0 `?` | 10 segments, 0 `?` |
| guitar / bass / piano notes | 892 / 379 / 905 | 881 / 387 / 843 |

The loop still does not appear. As the ruling asked, below are the first 12 chords and the bass runs for bars 1–12. Nothing was tuned.

Chords, bars 1–12 (bar start, name, bass, confidence):
```
 1  -2.186 Bbm    Bb 0.032     7  24.814 Dbsus2 Db 0.112
 2   2.314 Bbm    Bb 0.257     8  29.314 Ebm    Eb 0.009
 3   6.814 Dbsus2 Db 0.006     9  33.814 Bbm    Bb 0.101
 4  11.314 Ebsus2 Eb 0.026    10  38.314 Bbm    Bb 0.099
 5  15.814 Bbm    Bb 0.143    11  42.814 Db     Db 0.136
 6  20.314 Bbm    Bb 0.085    12  47.314 Ebm    Eb 0.016
```
Full chord list: Bbm Bbm Dbsus2 Ebsus2 Bbm Bbm Dbsus2 Ebm Bbm Bbm Db Ebm Bbm Bbm Db Ebm Bbm Bbm/Db Db Ebsus2 Bbm Bbm
Db Ebm Bbm Bbm Db Eb Bbm Bbm Db Ebsus2 Bbm Bbm Db Ebsus2 Bbm Bb Db Ebsus2 Bbm Bbm Db Ebsus2 Bb Ebm Ebsus2 Bbm Gdim.
The names repeat Bbm Bbm Db Ebm, which is the expected loop rotated and read through a mis-phased grid.

Bass root runs up to the end of bar 12 (start s, root, dur s):
```
 0.75 Bb 4.0    19.25 Bb 4.0    37.75 Bb 1.5    47.0  Eb 1.5
 5.5  Db 3.75   24.0  Db 4.5    40.25 Bb 1.5    49.25 Eb 1.75
10.0  Eb 4.25   28.5  Eb 3.75   42.5  Db 1.75   51.5  Bb 1.5
14.75 Bb 3.25   33.25 Bb 4.25   44.75 Db 1.5
18.0  F  1.0
```

### What goes wrong (diagnosis only)
- `bass_roots` quantizes run starts to 0.25 s, so single-bar root-change intervals come out as 4.50 or 4.75,
  never 4.64. Of the 53 intervals, 29 are kept by the ruling's |r − k| ≤ 0.25 test. Their iv/k are mostly
  4.50 (17 of 29), and the median is **4.50**. The mean is 4.478, and sum(iv)/sum(k) is 4.485.
- The first 8 bass changes (0.75 → 33.25 s over 7 bars) give exactly 4.643 s per bar, which matches the
  manual 4.64. Later intervals cluster at 4.50: either the band speeds up, or onset quantization biases them.
- With bar_len 4.50, the circular-mean phase lands at bar0 −2.186, about 1.6 s (a third of a bar) before the
  bass changes. For example, bar 3 starts at 6.81 while Db enters at 5.5. Every bar therefore straddles two
  chords, the confidences collapse, and `detect_loop` finds no repeat.
- For comparison, `fit_grid` with bar_len 4.64 gives bar0 −3.806 ≡ +0.83 s, in phase with the bass entry at 0.75 s.
