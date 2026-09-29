from __future__ import annotations
from pathlib import Path
from music_transcribe.cache import OutDir
from music_transcribe.schema import Tags, Harmony, LyricLine, Note, load_json
from music_transcribe.render.html import cifra_html, full_document
from music_transcribe.render.text import cifra_txt
from music_transcribe.render.midi import write_midi

PROGRAMS = {"guitar": 27, "bass": 33, "piano": 0}


def render_all(out: OutDir) -> dict[str, Path]:
    tags_path = out.stage("tags") / "tags.json"
    tags = load_json(Tags, tags_path) if tags_path.exists() else Tags()
    harmony = load_json(Harmony, out.stage("harmony") / "harmony.json")
    lp = out.stage("lyrics") / "lyrics.json"
    lyrics = load_json(LyricLine, lp) if lp.exists() else None   # None = not transcribed, [] = instrumental
    notes_by_inst = {p.stem: load_json(Note, p) for p in sorted(out.stage("notes").glob("*.json"))}
    r = out.stage("render")
    files = {"html": r / "cifra.html", "fragment": r / "cifra.fragment.html", "txt": r / "cifra.txt"}
    fragment = cifra_html(tags, harmony, lyrics, notes_by_inst)
    files["html"].write_text(full_document(fragment), encoding="utf-8")
    files["fragment"].write_text(fragment, encoding="utf-8")
    files["txt"].write_text(cifra_txt(tags, harmony, lyrics, notes_by_inst), encoding="utf-8")
    for inst, notes in notes_by_inst.items():
        p = r / f"{inst}.mid"
        write_midi(notes, p, PROGRAMS.get(inst, 0))
        files[inst] = p
    out.mark_done("render")
    return files
