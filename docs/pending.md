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

### Fix pass round 2 (2026-09-28, third run) — Status: PASSED

The controller's round-2 ruling targeted the remaining lyrics hallucination with two changes:

A. **`lyrics.transcribe` now uses `-ojf`** (output-json-full) instead of `-oj`, so whisper-cli's own per-token probabilities are written to the JSON and the existing confidence-based filters have real signal to work with (previously every hallucinated line fell back to the hardcoded 0.5 default because `-oj` doesn't include `tokens[].p`). Test updated: `tests/test_lyrics.py::test_transcribe_builds_command_and_reads_json` now asserts `"-ojf" in cmd`.
   - Side effect discovered while re-running the slow test: whisper.cpp's `-ojf` token dump can write a **raw, incomplete UTF-8 byte fragment** as an individual token's `"text"` (e.g. half of "♪"), because whisper's BPE tokens are only guaranteed valid UTF-8 once reassembled into the segment text, not individually. This made `json.loads(Path(...).read_text())` raise `UnicodeDecodeError` and crash the CLI. Fixed by decoding the file with `errors="replace"` before parsing (`Path(of + ".json").read_bytes().decode("utf-8", errors="replace")`). New regression test: `tests/test_lyrics.py::test_transcribe_tolerates_invalid_utf8_in_ojf_token_dump`.
B. **Vocal-activity energy gate** added: `lyrics.vocal_activity(y, sr, hop=512, threshold_db=-40.0, min_dur=0.3)` finds RMS-active regions relative to the stem's peak RMS; `lyrics.vocal_activity_for(path)` is the file-based wrapper; `lyrics.filter_by_activity(lines, regions, min_overlap=0.3)` drops whisper lines that don't overlap enough with active regions. `pipeline.stage_lyrics` now: computes regions on the 16 kHz vocals wav; if total active duration < 1.0 s, writes `lyrics.json = []` and skips whisper entirely (prints `"sem voz detectada"`); otherwise runs whisper and filters the result through `filter_by_activity`. New tests: `test_vocal_activity_finds_sine_burst_in_silence`, `test_vocal_activity_on_silence_is_empty`, `test_filter_by_activity_drops_lines_outside_active_regions`, and in `tests/test_cli.py` `test_lyrics_skips_whisper_when_no_vocal_activity` (asserts `transcribe`/`ensure_whisper_model` not called and `lyrics.json == []`) plus `vocal_activity_for` monkeypatched to `[(0.0, 100.0)]` in the three existing CLI tests that exercise the lyrics stage with a fake "RIFF" stub (so they don't hit real audio decoding).

Fast suite: `uv run pytest` → **103 passed, 1 deselected**, no warnings.

Slow run: `uv run pytest -m slow -v -s`, CPU, whisper `small.en` (cached), demucs `htdemucs_6s` (cached). **Runtime: 35.5 s.**

**Result: PASSED** (1 passed, 103 deselected).

- `bpm` 60.1, `key` "Bbm"
- `chords[:4]` = `loop` = `["Bbm", "Db", "Ebm", "Bbm"]`
- `lyrics.json` = `[]`
- guitar notes: 37, bass notes: 33
- `render/cifra.html` exists

What actually happened on the real run: the vocals stem still had enough demucs bleed/residual energy to clear the 1.0 s activity floor, so whisper did run — but with `-ojf` giving it real per-token confidence, it now transcribes the wordless track as `"♪♪♪"` (three segments) instead of fabricating a sentence. `parse_whisper_json`'s existing `NON_SPEECH`/no-letter-or-digit filters drop `"♪♪♪"` on their own, so `lyrics.json` comes out `[]` without even needing the new `filter_by_activity` step to intervene. Both round-2 fixes (real confidences via `-ojf`, and the activity gate as a backstop for genuinely silent tracks) are now in place and exercised.

`tests/test_integration.py` was not modified in either fix round — all its assertions pass as originally written in the brief (plus `--language en`).
