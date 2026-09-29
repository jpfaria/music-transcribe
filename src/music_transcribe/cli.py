from __future__ import annotations
from pathlib import Path
from typing import Callable
import typer
from music_transcribe import __version__, pipeline
from music_transcribe.cache import OutDir

app = typer.Typer(help="Letra, cifra e tab de uma música a partir do áudio, localmente.", no_args_is_help=True)


def _version(value: bool):
    if value:
        typer.echo(__version__); raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version, is_eager=True)):
    """music-transcribe"""


def _out(audio: Path, out: Path | None) -> OutDir:
    o = OutDir(out or audio.parent / f"{audio.stem}.transcribe")
    if not o.check_input(audio):
        typer.echo("áudio mudou, cache invalidado")
    return o


def _confirm(yes: bool):
    def f(name: str, mb: int) -> bool:
        return yes or typer.confirm(f"Baixar modelo {name} (~{mb} MB) para o cache local?")
    return f


def _parse_instruments(s: str) -> list[str]:
    return [i.strip() for i in s.split(",") if i.strip()]


TOOLS = ("ffmpeg", "ffprobe", "whisper-cli", "curl")


def _guarded(fn: Callable[[], None]) -> None:
    try:
        fn()
    except FileNotFoundError as e:
        named = f"{e.filename or ''} {e}"
        if any(t in named for t in TOOLS):
            typer.echo(f"ferramenta não encontrada ({e}). Instale com: brew install ffmpeg whisper-cpp", err=True)
        else:
            typer.echo(f"arquivo não encontrado: {e}", err=True)
        raise typer.Exit(1)
    except (RuntimeError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)


AUDIO = typer.Argument(..., exists=True, dir_okay=False)
OUT = typer.Option(None, "--out", help="Pasta de saída (padrão: <audio-sem-extensão>.transcribe/)")
FORCE = typer.Option(False, "--force", help="Refaz a etapa mesmo com cache")
YES = typer.Option(False, "--yes", help="Aceita baixar modelos (Whisper, pesos do demucs) sem perguntar")
DEVICE = typer.Option("mps", "--device", help="mps ou cpu (demucs; se mps falhar, tenta cpu uma vez)")
MODEL = typer.Option("large-v3", "--model", help="Modelo Whisper: large-v3, medium, medium.en, small.en")
LANGUAGE = typer.Option("auto", "--language", help="Idioma da letra: auto (detecta), en, pt, ...")
INSTRUMENTS = typer.Option("guitar,bass,piano", "--instruments", help="guitar, bass, piano, other (separados por vírgula)")


@app.command()
def run(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, yes: bool = YES, model: str = MODEL,
        instruments: str = INSTRUMENTS, device: str = DEVICE, language: str = LANGUAGE):
    """Roda o pipeline inteiro: tags → stems → lyrics → harmony → notes → render."""
    def body():
        o = _out(audio, out)
        c = _confirm(yes)
        typer.echo("→ tags")
        pipeline.stage_tags(audio, o, force)
        typer.echo("→ stems (demucs, pode levar minutos)")
        pipeline.stage_stems(audio, o, force, device, confirm=c)
        typer.echo("→ lyrics")
        try:
            pipeline.stage_lyrics(audio, o, model, c, force=force, language=language, device=device)
        except (RuntimeError, FileNotFoundError) as e:
            typer.echo(f"letra indisponível: {e}")
        typer.echo("→ harmony")
        pipeline.stage_harmony(audio, o, force, device, confirm=c)
        typer.echo("→ notes")
        pipeline.stage_notes(audio, o, _parse_instruments(instruments), force, device, confirm=c)
        typer.echo("→ render")
        files = pipeline.stage_render(o)
        for k, p in files.items():
            typer.echo(f"{k}: {p}")
    _guarded(body)


@app.command()
def tags(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE):
    _guarded(lambda: pipeline.stage_tags(audio, _out(audio, out), force))


@app.command()
def stems(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, yes: bool = YES, device: str = DEVICE):
    def body():
        for k, p in pipeline.stage_stems(audio, _out(audio, out), force, device, confirm=_confirm(yes)).items():
            typer.echo(f"{k}: {p}")
    _guarded(body)


@app.command()
def lyrics(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, yes: bool = YES, model: str = MODEL,
           language: str = LANGUAGE, device: str = DEVICE):
    _guarded(lambda: pipeline.stage_lyrics(audio, _out(audio, out), model, _confirm(yes), force=force,
                                           language=language, device=device))


@app.command()
def harmony(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, yes: bool = YES, device: str = DEVICE):
    def body():
        h = pipeline.stage_harmony(audio, _out(audio, out), force, device, confirm=_confirm(yes))
        typer.echo(f"{h.key} · {h.bpm} BPM · {h.meter} · loop {h.loop}")
    _guarded(body)


@app.command()
def notes(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, yes: bool = YES,
          instruments: str = INSTRUMENTS, device: str = DEVICE):
    _guarded(lambda: pipeline.stage_notes(audio, _out(audio, out), _parse_instruments(instruments), force, device,
                                          confirm=_confirm(yes)))


@app.command()
def render(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE):
    def body():
        for k, p in pipeline.stage_render(_out(audio, out)).items():
            typer.echo(f"{k}: {p}")
    _guarded(body)


if __name__ == "__main__":
    app()
