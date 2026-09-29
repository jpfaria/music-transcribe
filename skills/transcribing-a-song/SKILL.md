---
name: transcribing-a-song
description: Use when the user wants the lyrics, chords, cifra, tab, riff or solo of a song they have as a local audio file (flac, wav, mp3, m4a, aiff) — "me passa a cifra", "tira a letra", "tab do solo", "extrai do flac".
---

# Transcribing a song from audio

## Overview
The user has the audio and wants to play along. The `music-transcribe` CLI (this plugin) does it all locally: stems, lyrics (Whisper), key/tempo/chords per bar, notes with articulations, and a cifra page. Your job: run it, **review its JSON**, deliver. Never answer from memory or with another tool.

Not for songs the user has no file of (no downloading audio).

## Steps
1. **CLI.** Use `music-transcribe` if on PATH; else `uv run --project <plugin root> music-transcribe` (plugin root = this skill's base directory, two levels up). `ffmpeg` or `whisper-cli` missing → give `brew install ffmpeg whisper-cpp` and stop.
2. **Run** `music-transcribe run "<audio>" --yes` in the background and wait for it (minutes). If `~/.cache/music-transcribe/ggml-large-v3.bin` is absent, first say in one line that it downloads Whisper large-v3 (~3 GB), then run. Output: `<audio dir>/<audio name without extension>.transcribe/`.
3. **Review** before replying (read the files, don't trust the page):
   - `harmony/harmony.json`: `bpm` plausible for the style (slow blues 40–70, pop 80–130), `meter`, `key`, `loop`. Implausible tempo, or empty `loop` on a clearly repeating song → `harmony "<audio>" --force`, then `notes "<audio>" --force`, then `render "<audio>"` (stages are cached; a plain `run` keeps stale notes). Chords with `confidence < 0.3` are uncertain.
   - `lyrics/lyrics.json`: count lines with `confidence < 0.6` (shown with `?`). Whisper runs in English only; for a song in another language say the lyrics are unreliable. `tags/tags.json` `embedded_lyrics`, if present, is the file's own text: mention it.
   - `notes/<inst>.json`: bar = `floor((start - bar0) / bar_len) + 1`. Count bars with any `low` note and the `reason`s (`bleed`, `weak`, `unstable`). Bars where **every** note is `low` = probably accompaniment/bleed, not the part.
4. **Publish** `render/cifra.fragment.html` as an artifact (load `artifact-design` first).
5. **Copy** the whole output folder to the user's transcription folder. João: `~/Library/Mobile Documents/com~apple~CloudDocs/Musica/Transcricoes/<Artista - Título>/` (from `tags.json`; file name if empty).
6. **Reply, 4 lines max:**
   1. key · BPM · meter · loop
   2. lyrics: N uncertain lines (`?`) of M, plus language caveat if any
   3. tab: N bars with low notes (reasons); bars that look like bleed
   4. artifact link · copied folder path

## Rules
- Confidence is shown, never hidden: no `low` note, `?` line or chord under 0.3 presented as certain.
- No audio leaves the machine. No online lyric/chord lookup unless the user asks. Never "fix" a lyric or chord from memory; call it uncertain.
- Do not skip stages to save time: the guitarist wants the tab too.

| Temptation | Reality |
|---|---|
| "I know this song, I'll write the chords" | Unchecked against his recording (key, capo, version). Run the CLI. |
| "Install Chordino/madmom instead" | The pipeline already does it, with confidence. |
| "Lyrics only / chords only is enough" | `run` does all; the page needs all. |
| "Local path is enough" | Publish the artifact and copy the folder. |
