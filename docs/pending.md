# Pendências (state at the end of the final fix wave, 2026-09-29)

## What passes
- Fast suite: `uv run pytest -q`, 139 passed, warning-free. Unit tests never touch the real model
  caches (autouse fixture in `tests/conftest.py`).
- Slow suite: `uv run pytest -m slow -v -s` (~40 s, CPU, whisper small.en). bpm 60.1, key Bbm,
  chords/loop Bbm Db Ebm Bbm, bass pitch classes = roots, every lead pitch found in guitar ∪ other.
  The instrumental-lyrics assertion is flaky (see below).
- Real song: see `docs/real-run.md`.

## Known limits
- Guitar and bass are monophonic lines. For simultaneous notes the loudest is kept, marked `medium`/`chord`.
  Piano keeps simultaneous notes.
- The vocal-activity gate is −40 dB relative to the vocals stem's own peak. On a near-silent stem
  (instrumental; demucs residue peaks at about −62 dBFS) the whole track counts as active, so whisper can
  hallucinate. demucs applies a random time shift, so the residue, and therefore the lyrics, change from run
  to run: on the synthetic fixture, 4 of 8 runs on the new code and 4 of 7 on the pre-wave code (ef9b851)
  gave non-empty lyrics. Whisper itself is deterministic on a fixed wav.
- `--language auto` can misdetect on short vocal passages.
- Chord confidence on dense real mixes is mostly < 0.3, and the loop may not be detected (real run).

## Follow-ups after merge
The review's own "after merge" triage list was not in the files given to the fix wave. This list is
`.superpowers/sdd/.../deferred-minors.md` minus what the wave fixed, plus the wave's findings.
1. Lyrics gate: add an absolute floor (e.g. vocals peak < −50 dBFS → no voice) or run demucs with
   `--shifts 0` in tests, so the slow test's instrumental assertion becomes deterministic.
2. Bass `chord` flags: ignore octave/12th doubles when deciding that a note suppressed a chord tone.
3. Chords on real mixes: sus2/sus4 overtake triads. Revisit the template weighting with the real-run data.
4. Equal-amplitude onset ties both survive the monophonic reduction (`notes/detect.py`).
5. `schema._from_dict` lacks Optional[dataclass] handling; no round-trip tests for LyricLine/Tags/pc_name.
6. On a partial demucs failure (missing stem), the stems already moved are left in `stems/`.
7. `estimate_tempo` computes the tempogram twice in the < 3 changes branch, and single-linkage clustering
   can drift. `bass_roots` bin loop is O(n²).
8. `_gap_voiced` runs pyin per gapped pair (speed).
9. `bar_tab` has no bar_len ≤ 0 guard (only `bars_for` has one). `cifra_txt` scans notes per bar
   (O(bars×notes)). Thresholds 0.3/0.6 are magic numbers.
10. Unused `--accent-ink` CSS token.
11. `run --force` re-runs demucs (a literal reading of the spec). Consider `--force-from <stage>`.
