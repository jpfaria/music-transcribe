from __future__ import annotations
from music_transcribe.schema import Tags, Harmony, LyricLine, Note
from music_transcribe.render.tab import bars_for, STRING_NAMES


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def cifra_txt(tags: Tags, harmony: Harmony, lyrics: list[LyricLine], notes_by_inst: dict[str, list[Note]]) -> str:
    out = [f"{tags.title or 'Sem título'} — {tags.artist or 'Artista desconhecido'}",
           f"Tom: {harmony.key} | {harmony.bpm:.0f} BPM em {harmony.meter} | compasso = {harmony.bar_len:.2f} s", ""]
    if harmony.loop:
        out += ["PROGRESSÃO (loop): | " + " | ".join(harmony.loop) + " |", ""]
    out.append("ACORDES POR COMPASSO")
    for c in harmony.chords:
        flag = "" if c.confidence >= 0.3 else "  (?)"
        out.append(f"{c.bar:4d} {_mmss(c.start)}  {c.name}{flag}")
    out += ["", "LETRA"]
    if not lyrics:
        out.append("(instrumental)")
    for l in lyrics:
        out.append(f"[{_mmss(l.start)}] {l.text}{'' if l.confidence >= 0.6 else ' (?)'}")
    for inst, notes in notes_by_inst.items():
        if inst not in STRING_NAMES:
            continue
        out += ["", f"TAB — {inst.upper()}  (b=bend, br=bend-release, ~=vibrato, /=slide, h=hammer, p=pull)"]
        for bar, tab in bars_for(notes, harmony, inst):
            chord = next((c.name for c in harmony.chords if c.bar == bar), "")
            low = sum(1 for n in notes if n.confidence == "low" and harmony.bar0 + (bar - 1) * harmony.bar_len <= n.start < harmony.bar0 + bar * harmony.bar_len)
            out += [f"compasso {bar}  {chord}" + (f"  [{low} nota(s) confiança baixa]" if low else ""), tab, ""]
    return "\n".join(out) + "\n"
