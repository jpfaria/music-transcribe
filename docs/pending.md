# Pendências (state at the end of the final fix wave, 2026-09-29)

## What passes
- Fast suite: `uv run pytest -q`, 149 passed, warning-free. Unit tests never touch the real model
  caches (autouse fixture in `tests/conftest.py`).
- Slow suite: `uv run pytest -m slow -v -s` (CPU, whisper small.en, demucs `--shifts 0`): 3 consecutive
  passes in 25.8 s, 29.6 s and 27.0 s; after the bass-step/least-squares change, 1 more pass in 26.6 s; after the family loop/confidence change, 1 more pass in 25.5 s; after the sus-family change, 1 more pass in 24.4 s. Results: bpm 60.1, key Bbm, chords/loop Bbm Db Ebm Bbm, lyrics `[]`,
  bass pitch classes = roots, every lead pitch found in guitar ∪ other.
- Real song: see `docs/real-run.md`.

## Known limits
- Guitar and bass are monophonic lines. For simultaneous notes the loudest is kept, marked `medium`/`chord`.
  Piano keeps simultaneous notes.
- The vocal-activity gate is −40 dB relative to the vocals stem's peak, with an absolute floor: a stem whose
  peak RMS stays under −45 dBFS has no vocals.
- `--language auto` can misdetect on short vocal passages.
- Real song (`docs/real-run.md`, run 5):
  - bar_len 4.53 s is the song's average (the intro is 4.63 s, later bars about 4.54 s).
  - Loop detected: Bbm | Bbm | Db | Ebsus2, i.e. the expected cycle started at the intro pickup. At the Eb
    position the name vote is tied (Ebsus2 4, Ebm 4), and the first seen name wins.
  - 34 of 49 chords are under 0.3.
  - A single global bar_len cannot follow tempo drift.

## Follow-ups after merge
The review's own "after merge" triage list was not in the files given to the fix wave. This list is
`.superpowers/sdd/.../deferred-minors.md` minus what the wave fixed, plus the wave's findings.
1. Bass `chord` flags: ignore octave/12th doubles when deciding that a note suppressed a chord tone.
2. Chords on real mixes: sus2/sus4 overtake triads. Revisit the template weighting with the real-run data.
3. Equal-amplitude onset ties both survive the monophonic reduction (`notes/detect.py`).
4. `schema._from_dict` lacks Optional[dataclass] handling; no round-trip tests for LyricLine/Tags/pc_name.
5. On a partial demucs failure (missing stem), the stems already moved are left in `stems/`.
6. `estimate_tempo` computes the tempogram twice in the < 3 changes branch, and single-linkage clustering
   can drift. `bass_roots` bin loop is O(n²).
7. `_gap_voiced` runs pyin per gapped pair (speed).
8. `bar_tab` has no bar_len ≤ 0 guard (only `bars_for` has one). `cifra_txt` scans notes per bar
   (O(bars×notes)). Thresholds 0.3/0.6 are magic numbers.
9. Unused `--accent-ink` CSS token.
10. `run --force` re-runs demucs (a literal reading of the spec). Consider `--force-from <stage>`.

## Residuais da re-review final (2026-09-29) — pós-merge

- `render_all` renderiza todo `notes/*.json`, inclusive JSON antigo de instrumento não pedido ou invalidado. Renderizar só os instrumentos com `.done.<inst>` ou apagar o JSON na invalidação.
- `refine_bar_len`: `k` sem teto; um gap ≥ ~19 compassos com erro de 4% no tempograma pode deslocar o índice. Limitar `k` (ex.: ≤ 8) ou iterar o ajuste.
- `detect_loop`: empate resolvido por ordem de aparição (Ebsus2 × Ebm 4–4). Preferir a tríade sobre sus, ou mostrar "Ebm/Ebsus2".
- Flag `chord` no baixo dispara em dobras de oitava; ignorar notas simultâneas a uma oitava.
- `fit_grid`: para 0 < bar0 ≤ 5% do compasso, [0, bar0) fica fora do compasso 1; alinhar docstring e comportamento.
- `ensure_demucs_weights`: symlink quebrado passa como "encontrado"; validar o alvo.
- `check_input`: pasta de saída anterior a esta versão (sem `input.sha1`) é adotada sem invalidar.
- SKILL.md: linha "um `run` simples mantém notas velhas" está desatualizada (harmony recomputado invalida notes); lista de `reason` não cita `chord`.
- Rodar uma música real com `large-v3` (só `small.en` foi medido).
- Calibrar o corte 0.3 de confiança de acorde com mais músicas (hoje é heurístico: margem × 10).
