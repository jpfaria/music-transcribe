from __future__ import annotations
from html import escape
from music_transcribe.schema import Tags, Harmony, LyricLine, Note, pitch_name
from music_transcribe.render.chord_diagram import voicing_for, svg
from music_transcribe.render.tab import bars_for

CSS = """
:root{--bg:#f6f3ee;--ink:#1c1a1f;--muted:#6b6470;--rule:#dcd6cc;--panel:#fbf9f5;--accent:#b8651a;--accent-ink:#7a4110;--chord:#2a5f8a;--pre:#fffdf9;--low:#b3261e;--mid:#8a6d00}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15131a;--ink:#ece7df;--muted:#9d95a5;--rule:#2f2b37;--panel:#1d1a24;--accent:#e08a3a;--accent-ink:#f2b06d;--chord:#8fc1ec;--pre:#100f14;--low:#ff8a80;--mid:#e6c34a}}
:root[data-theme="dark"]{--bg:#15131a;--ink:#ece7df;--muted:#9d95a5;--rule:#2f2b37;--panel:#1d1a24;--accent:#e08a3a;--accent-ink:#f2b06d;--chord:#8fc1ec;--pre:#100f14;--low:#ff8a80;--mid:#e6c34a}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:16px;line-height:1.5;padding-inline:clamp(16px,4vw,48px);padding-block:32px 64px}
.wrap{max-width:1040px;margin:0 auto;display:grid;gap:40px}
h1,h2{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;text-wrap:balance;margin:0}
h1{font-size:clamp(34px,6vw,58px);line-height:1;font-weight:700;letter-spacing:-.02em}
h2{font-size:22px;font-weight:700;padding-bottom:8px;border-bottom:2px solid var(--rule);margin-bottom:16px}
.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
header{display:grid;gap:10px}
.meta{display:flex;flex-wrap:wrap;gap:8px 20px;font-family:"IBM Plex Mono",monospace;font-size:14px;color:var(--muted)}
.meta b{color:var(--ink);font-weight:500}
.prog{display:flex;flex-wrap:wrap;gap:8px;font-family:"IBM Plex Mono",monospace;font-size:18px}
.prog span{background:var(--panel);border:1px solid var(--rule);padding:8px 14px;border-radius:4px;font-weight:500;color:var(--chord)}
.chords-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;max-width:760px}
.cd{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:12px 10px 8px;display:grid;justify-items:center;gap:4px}
.cd .nm{font-family:"IBM Plex Mono",monospace;font-size:20px;font-weight:500;color:var(--chord)}
.cd svg{width:120px;height:auto;max-width:100%}
.cd .st{stroke:var(--ink)}.cd .dot{fill:var(--ink)}.cd .lbl{fill:var(--muted);font:11px "IBM Plex Mono",monospace}.cd .nut{stroke:var(--ink);stroke-width:4}.cd .brr{fill:var(--ink);opacity:.9}
.cols{display:grid;grid-template-columns:1fr;gap:32px}
@media (min-width:760px){.cols{grid-template-columns:1fr 1fr}}
.line{display:grid;grid-template-columns:auto 1fr;gap:12px;align-items:baseline;padding:4px 0}
.line time{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted)}
.line.low{color:var(--low)} .line.low::after{content:" ?";color:var(--low)}
.note{background:var(--panel);border-left:3px solid var(--accent);padding:12px 16px;font-size:14px;color:var(--muted);max-width:60em}
.bars{display:grid;grid-template-columns:1fr;gap:18px}
@media (min-width:760px){.bars{grid-template-columns:1fr 1fr}}
.bar{margin:0;background:var(--panel);border:1px solid var(--rule);border-radius:4px;overflow:hidden}
.bar figcaption{display:flex;gap:12px;align-items:baseline;padding:8px 12px;border-bottom:1px solid var(--rule);font-family:"IBM Plex Mono",monospace;font-size:13px}
.bar .chord{color:var(--chord);font-weight:500}.bar .time{margin-left:auto;color:var(--muted)}
.bar pre{margin:0;padding:10px 12px;background:var(--pre);font-family:"IBM Plex Mono",monospace;font-size:13.5px;line-height:1.35;overflow-x:auto;color:var(--ink)}
.conf-low{color:var(--low)}.conf-medium{color:var(--mid)}
.legend{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--muted);margin-bottom:14px}
.chordtag{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--chord)}
"""
FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">'


def _mmss(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _chord_cards(names: list[str]) -> str:
    cards = []
    for n in names:
        v = voicing_for(n)
        body = svg(v, n) if v else '<div class="legend">sem diagrama</div>'
        cards.append(f'<div class="cd"><div class="nm">{escape(n)}</div>{body}</div>')
    return '<div class="chords-grid">' + "".join(cards) + "</div>"


def _chord_at(harmony: Harmony, t: float) -> str:
    for c in reversed(harmony.chords):
        if c.start <= t:
            return c.name
    return ""


def _lyrics(harmony: Harmony, lyrics: list[LyricLine] | None) -> str:
    if lyrics is None:
        return '<p class="legend">Letra não transcrita (etapa de letra indisponível nesta execução).</p>'
    if not lyrics:
        return '<p class="legend">Instrumental — nenhuma voz detectada.</p>'
    rows = []
    for l in lyrics:
        cls = "line low" if l.confidence < 0.6 else "line"
        rows.append(f'<div class="{cls}"><time>{_mmss(l.start)}</time><div><span class="chordtag">{escape(_chord_at(harmony, l.start))}</span><br>{escape(l.text)}</div></div>')
    return "".join(rows)


def _tab_section(inst: str, notes: list[Note], harmony: Harmony) -> str:
    bars = bars_for(notes, harmony, inst)
    if not bars:
        return ""
    figs = []
    for bar, tab in bars:
        t0 = harmony.bar0 + (bar - 1) * harmony.bar_len
        chord = next((c.name for c in harmony.chords if c.bar == bar), "")
        in_bar = [n for n in notes if t0 <= n.start < t0 + harmony.bar_len]
        worst = "low" if any(n.confidence == "low" for n in in_bar) else ("medium" if any(n.confidence == "medium" for n in in_bar) else "high")
        reasons = sorted({n.reason for n in in_bar if n.reason})
        figs.append(f'<figure class="bar"><figcaption><span>Compasso {bar}</span><span class="chord">{escape(chord)}</span>'
                    f'<span class="conf-{worst}">{worst}{" · " + ", ".join(reasons) if reasons else ""}</span><span class="time">{_mmss(t0)}</span></figcaption>'
                    f'<pre>{escape(tab)}</pre></figure>')
    return (f'<section><h2>{escape(inst.capitalize())} · tab</h2>'
            f'<div class="legend">{harmony.meter} · cada "|" é um tempo · b=bend (casa alvo) · br=bend-release · ~=vibrato · /=slide · h=hammer · p=pull · cor = confiança</div>'
            f'<div class="bars">{"".join(figs)}</div></section>')


def _head_parts(title: str) -> list[str]:
    return [f"<title>{escape(title)}</title>", FONTS, f"<style>{CSS}</style>"]


def _body_parts(tags: Tags, harmony: Harmony, lyrics: list[LyricLine] | None, notes_by_inst: dict[str, list[Note]]) -> list[str]:
    title = tags.title or "Sem título"
    uniq = list(dict.fromkeys(harmony.loop or [c.name for c in harmony.chords]))
    low_chords = sum(1 for c in harmony.chords if c.confidence < 0.3)
    parts = ['<div class="wrap">',
             "<header>", f'<div class="eyebrow">{escape(tags.artist)}{" · " + escape(tags.album) if tags.album else ""}</div>',
             f"<h1>{escape(title)}</h1>",
             f'<div class="meta"><span>Tom <b>{escape(harmony.key)}</b></span><span>Andamento <b>{harmony.bpm:.0f} BPM · {harmony.meter}</b></span><span>Compasso <b>{harmony.bar_len:.2f} s</b></span></div>']
    if harmony.loop:
        parts.append('<div class="prog">' + "".join(f"<span>{escape(n)}</span>" for n in harmony.loop) + "</div>")
    parts.append(_chord_cards(uniq))
    parts.append(f'<div class="note">Acordes por compasso: {len(harmony.chords)}; {low_chords} com confiança baixa (marcados na lista).</div></header>')
    parts.append('<div class="cols"><section><h2>Letra</h2>' + _lyrics(harmony, lyrics) + "</section>")
    parts.append('<section><h2>Acordes por compasso</h2><div class="legend">' + " ".join(
        f'<span class="{"conf-low" if c.confidence < 0.3 else ""}">{c.bar}:{escape(c.name)}</span>' for c in harmony.chords) + "</div></section></div>")
    for inst in ("guitar", "bass"):
        if inst in notes_by_inst:
            parts.append(_tab_section(inst, notes_by_inst[inst], harmony))
    if "piano" in notes_by_inst and notes_by_inst["piano"]:
        rows = "".join(f"<div class='line conf-{n.confidence}'><time>{_mmss(n.start)}</time><div>{escape(pitch_name(n.pitch))}</div></div>" for n in notes_by_inst["piano"])
        parts.append(f"<section><h2>Piano · notas</h2>{rows}</section>")
    parts.append("</div>")
    return parts


def cifra_html(tags: Tags, harmony: Harmony, lyrics: list[LyricLine] | None, notes_by_inst: dict[str, list[Note]]) -> str:
    title = tags.title or "Sem título"
    parts = _head_parts(title) + _body_parts(tags, harmony, lyrics, notes_by_inst)
    return "\n".join(parts)


def full_document(fragment: str, lang: str = "pt-BR") -> str:
    """Wrap the cifra_html() fragment into a complete, self-contained HTML document for disk output."""
    split_at = fragment.index('<div class="wrap">')
    head, body = fragment[:split_at].rstrip("\n"), fragment[split_at:]
    return (f'<!doctype html>\n<html lang="{lang}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            f"{head}</head><body>{body}</body></html>")
