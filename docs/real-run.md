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
