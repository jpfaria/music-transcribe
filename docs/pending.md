# Pendências

## Task 17 — synthetic end-to-end integration test (`tests/test_integration.py`)

### Fix pass (2026-09-28, second run)

Fixed and unit-tested per controller ruling:

1. **Phantom pickup bar (`harmony/grid.py::fit_grid`)** — fixed. After the existing "shift bar0 to at-or-before the first change" step, a pickup ≥ 75% of a bar is now treated as a phase-wrap artifact and folded back (`bar0 += bar_len`). New test: `tests/test_grid.py::test_fit_grid_no_phantom_bar_when_song_starts_on_downbeat`. Confirmed on the real run: `bar0` is now `0.027` (was `-3.967`).
2. **Chord misrooting from overtones (`harmony/chords.py::chord_for_bar`)** — fixed. `+0.10` when `bass_pc == root` (bass is the strongest root evidence), `-0.03` for sus2/sus4 (rarer). New test: `tests/test_chords.py::test_root_position_beats_sus_reinterpretation`. Confirmed on the real run: bar 2 is now named `"Db"` (was `"Absus4/Db"`); all 4 pre-existing chord tests (Bbm, F7, Db/F, Ebm7b5) still pass unchanged.
3. **Lyrics hallucination filtering (`lyrics.py::parse_whisper_json`)** — partially fixed. Segments with no letter/digit (`". . ."`, `"..."`, `"♪ ♪"`) are dropped, and a segment repeating the previous kept segment's text at confidence < 0.6 twice in a row is deduped to the first occurrence. New tests: `tests/test_lyrics.py::test_parse_drops_punctuation_only_hallucinations`, `::test_parse_drops_repeated_low_confidence_hallucination`, `::test_parse_keeps_normal_line_unaffected`.

Fast suite after the fixes: `uv run pytest` → **98 passed, 1 deselected**, no warnings.

### Slow run after the fixes: `uv run pytest -m slow -v -s`, CPU, whisper `small.en` (cached from the prior run), demucs `htdemucs_6s` (cached). Runtime: **32 s**.

**Result: still FAILS**, on the lyrics assertion only. Harmony is now fully correct:

- `bpm` 60.1, `key` "Bbm" — PASS
- `chords[:4]` = `["Bbm", "Db", "Ebm", "Bbm"]` — **matches exactly**
- `loop` = `["Bbm", "Db", "Ebm", "Bbm"]` — **matches exactly**
- `render/cifra.html` exists — PASS

`lyrics.json` is not `[]`:

```json
[{"start": 0.0, "end": 3.0, "text": "This is a song from the beginning of the song.", "confidence": 0.5}]
```

**Diagnosis:** this is a different whisper failure mode than the one the controller's ruling targeted. It is a single segment, occurring once (not repeated, so the repeat-dedup rule doesn't apply), containing real words (so the letter/digit check doesn't apply) — whisper `small.en`, prompted with `"Song lyrics:"` on a wordless track, fabricated one plausible-sounding English sentence covering the first 3 s, at confidence 0.5. This is a known one-shot hallucination pattern (distinct from the repeated-token or punctuation-only patterns already handled) that would need a different heuristic to catch — e.g. treating a single low-confidence segment spanning most/all of a silent-vocals track as suspect, or cross-checking against a voice-activity-detection pass on the vocals stem — none of which was in the controller's ruling, so it was not implemented here to avoid inventing an unreviewed heuristic that could also suppress real quiet/low-confidence lyrics.

The test assertions were not weakened. `tests/test_integration.py` is committed as specified; the harmony/loop portion is now a real passing regression check, and the lyrics portion documents a known, reproducible, distinct gap for follow-up.

### Follow-up (not implemented)

- `parse_whisper_json` has no defense against a single, non-repeated, well-formed hallucinated sentence on wordless audio. Needs a design decision (VAD cross-check, or a "whole track is one low-confidence segment" heuristic) beyond the scope of the three rulings above.
