from __future__ import annotations
from pathlib import Path
from typing import Callable
import typer
from music_transcribe.cache import OutDir
from music_transcribe.schema import save_json, load_json, Harmony
from music_transcribe.tags import read_tags
from music_transcribe.stems import separate, STEM_NAMES
from music_transcribe.audio import to_wav
from music_transcribe.models import ensure_whisper_model
from music_transcribe.lyrics import transcribe, vocal_activity_for, filter_by_activity
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


def stage_stems(audio: Path, out: OutDir, force: bool = False, device: str = "mps",
                confirm: Callable[[str, int], bool] | None = None) -> dict[str, Path]:
    d = out.stage("stems")
    if not _skip(out, "stems", force):
        wav = d / "_input.wav"
        to_wav(audio, wav)
        separate(wav, d, device=device, confirm=confirm)
        wav.unlink(missing_ok=True)
        out.mark_done("stems")
        out.invalidate("lyrics", "harmony", "notes")
    return {s: d / f"{s}.wav" for s in STEM_NAMES}


def stage_lyrics(audio: Path, out: OutDir, model: str, confirm: Callable[[str, int], bool], force: bool = False,
                  language: str = "auto", device: str = "mps") -> None:
    if _skip(out, "lyrics", force):
        return
    stems = stage_stems(audio, out, device=device, confirm=confirm)
    d = out.stage("lyrics")
    (d / "lyrics.json").unlink(missing_ok=True)   # a failed run must not leave stale lyrics behind
    v16 = to_wav(stems["vocals"], d / "vocals16.wav", sr=16000, mono=True)
    regions = vocal_activity_for(v16)
    total_active = sum(e - s for s, e in regions)
    if total_active < 1.0:
        typer.echo("sem voz detectada")
        save_json([], d / "lyrics.json")
        out.mark_done("lyrics")
        return
    fallback = "medium.en" if language == "en" else "medium"
    mp = ensure_whisper_model(model, confirm)
    if mp is None and model != fallback:
        mp = ensure_whisper_model(fallback, confirm)
    if mp is None:
        raise RuntimeError("nenhum modelo Whisper disponível; rode com --yes ou baixe manualmente")
    lines = filter_by_activity(transcribe(v16, mp, language=language), regions)
    save_json(lines, d / "lyrics.json")
    out.mark_done("lyrics")


def stage_harmony(audio: Path, out: OutDir, force: bool = False, device: str = "mps",
                  confirm: Callable[[str, int], bool] | None = None) -> Harmony:
    d = out.stage("harmony")
    if _skip(out, "harmony", force):
        return load_json(Harmony, d / "harmony.json")
    stems = stage_stems(audio, out, device=device, confirm=confirm)
    h = analyze(stems)
    save_json(h, d / "harmony.json")
    out.mark_done("harmony")
    out.invalidate("notes")
    return h


def stage_notes(audio: Path, out: OutDir, instruments: list[str], force: bool = False, device: str = "mps",
                confirm: Callable[[str, int], bool] | None = None) -> None:
    for inst in instruments:
        if inst not in ALLOWED_INSTRUMENTS:
            raise ValueError(f"instrumento desconhecido: {inst!r} (use guitar, bass, piano, other)")
    todo = [i for i in instruments if force or not out.done("notes", i)]
    if not todo:
        return
    stems = stage_stems(audio, out, device=device, confirm=confirm)
    h = stage_harmony(audio, out, device=device, confirm=confirm)
    d = out.stage("notes")
    for inst in todo:
        save_json(transcribe_instrument(stems[inst], inst, h), d / f"{inst}.json")
        out.mark_done("notes", inst)


def stage_render(out: OutDir) -> dict[str, Path]:
    return render_all(out)
