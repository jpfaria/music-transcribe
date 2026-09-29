from __future__ import annotations
from pathlib import Path
from typing import Callable
from music_transcribe.cache import OutDir
from music_transcribe.schema import save_json, load_json, Harmony
from music_transcribe.tags import read_tags
from music_transcribe.stems import separate, STEM_NAMES
from music_transcribe.audio import to_wav
from music_transcribe.models import ensure_whisper_model
from music_transcribe.lyrics import transcribe
from music_transcribe.harmony import analyze
from music_transcribe.notes import transcribe_instrument
from music_transcribe.render import render_all

ALLOWED_INSTRUMENTS = [s for s in STEM_NAMES if s not in ("drums", "vocals")]


def _skip(out: OutDir, name: str, force: bool) -> bool:
    return out.done(name) and not force


def stage_tags(audio: Path, out: OutDir, force: bool = False) -> None:
    if _skip(out, "tags", force):
        return
    save_json(read_tags(audio), out.stage("tags") / "tags.json")
    out.mark_done("tags")


def stage_stems(audio: Path, out: OutDir, force: bool = False, device: str = "mps") -> dict[str, Path]:
    d = out.stage("stems")
    if not _skip(out, "stems", force):
        wav = d / "_input.wav"
        to_wav(audio, wav)
        separate(wav, d, device=device)
        wav.unlink(missing_ok=True)
        out.mark_done("stems")
    return {s: d / f"{s}.wav" for s in STEM_NAMES}


def stage_lyrics(audio: Path, out: OutDir, model: str, confirm: Callable[[str, int], bool], force: bool = False,
                  language: str = "auto") -> None:
    if _skip(out, "lyrics", force):
        return
    stems = stage_stems(audio, out)
    d = out.stage("lyrics")
    v16 = to_wav(stems["vocals"], d / "vocals16.wav", sr=16000, mono=True)
    fallback = "medium.en" if language == "en" else "medium"
    mp = ensure_whisper_model(model, confirm)
    if mp is None and model != fallback:
        mp = ensure_whisper_model(fallback, confirm)
    if mp is None:
        raise RuntimeError("nenhum modelo Whisper disponível; rode com --yes ou baixe manualmente")
    save_json(transcribe(v16, mp, language=language), d / "lyrics.json")
    out.mark_done("lyrics")


def stage_harmony(audio: Path, out: OutDir, force: bool = False) -> Harmony:
    d = out.stage("harmony")
    if _skip(out, "harmony", force):
        return load_json(Harmony, d / "harmony.json")
    stems = stage_stems(audio, out)
    h = analyze(stems)
    save_json(h, d / "harmony.json")
    out.mark_done("harmony")
    return h


def stage_notes(audio: Path, out: OutDir, instruments: list[str], force: bool = False) -> None:
    for inst in instruments:
        if inst not in ALLOWED_INSTRUMENTS:
            raise ValueError(f"instrumento desconhecido: {inst!r} (use guitar, bass, piano, other)")
    if _skip(out, "notes", force):
        return
    stems = stage_stems(audio, out)
    h = stage_harmony(audio, out)
    d = out.stage("notes")
    for inst in instruments:
        save_json(transcribe_instrument(stems[inst], inst, h), d / f"{inst}.json")
    out.mark_done("notes")


def stage_render(out: OutDir) -> dict[str, Path]:
    return render_all(out)
