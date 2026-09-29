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
    return OutDir(out or audio.parent / f"{audio.stem}.transcribe")


def _confirm(yes: bool):
    def f(name: str, mb: int) -> bool:
        return yes or typer.confirm(f"Baixar modelo Whisper {name} (~{mb} MB) para o cache local?")
    return f


def _parse_instruments(s: str) -> list[str]:
    return [i.strip() for i in s.split(",") if i.strip()]


def _guarded(fn: Callable[[], None]) -> None:
    try:
        fn()
    except FileNotFoundError as e:
        typer.echo(
            f"ferramenta não encontrada (ffmpeg/whisper-cli?). Instale com: brew install ffmpeg whisper-cpp ({e})",
            err=True,
        )
        raise typer.Exit(1)
    except (RuntimeError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)


AUDIO = typer.Argument(..., exists=True, dir_okay=False)
OUT = typer.Option(None, "--out", help="Pasta de saída (padrão: <audio-sem-extensão>.transcribe/)")
FORCE = typer.Option(False, "--force", help="Refaz a etapa mesmo com cache")


@app.command()
def run(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, yes: bool = typer.Option(False, "--yes"),
        model: str = typer.Option("large-v3", "--model"), instruments: str = typer.Option("guitar,bass,piano", "--instruments"),
        device: str = typer.Option("mps", "--device"), language: str = typer.Option("auto", "--language")):
    """Roda o pipeline inteiro: tags → stems → lyrics → harmony → notes → render."""
    def body():
        o = _out(audio, out)
        typer.echo("→ tags")
        pipeline.stage_tags(audio, o, force)
        typer.echo("→ stems (demucs, pode levar minutos)")
        pipeline.stage_stems(audio, o, force, device)
        typer.echo("→ lyrics")
        pipeline.stage_lyrics(audio, o, model, _confirm(yes), force=force, language=language)
        typer.echo("→ harmony")
        pipeline.stage_harmony(audio, o, force)
        typer.echo("→ notes")
        pipeline.stage_notes(audio, o, _parse_instruments(instruments), force)
        typer.echo("→ render")
        files = pipeline.stage_render(o)
        for k, p in files.items():
            typer.echo(f"{k}: {p}")
    _guarded(body)


@app.command()
def tags(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE):
    _guarded(lambda: pipeline.stage_tags(audio, _out(audio, out), force))


@app.command()
def stems(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, device: str = typer.Option("mps", "--device")):
    def body():
        for k, p in pipeline.stage_stems(audio, _out(audio, out), force, device).items():
            typer.echo(f"{k}: {p}")
    _guarded(body)


@app.command()
def lyrics(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, yes: bool = typer.Option(False, "--yes"),
           model: str = typer.Option("large-v3", "--model"), language: str = typer.Option("auto", "--language")):
    _guarded(lambda: pipeline.stage_lyrics(audio, _out(audio, out), model, _confirm(yes), force=force, language=language))


@app.command()
def harmony(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE):
    def body():
        h = pipeline.stage_harmony(audio, _out(audio, out), force)
        typer.echo(f"{h.key} · {h.bpm} BPM · {h.meter} · loop {h.loop}")
    _guarded(body)


@app.command()
def notes(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE,
          instruments: str = typer.Option("guitar,bass,piano", "--instruments")):
    _guarded(lambda: pipeline.stage_notes(audio, _out(audio, out), _parse_instruments(instruments), force))


@app.command()
def render(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE):
    def body():
        for k, p in pipeline.stage_render(_out(audio, out)).items():
            typer.echo(f"{k}: {p}")
    _guarded(body)


if __name__ == "__main__":
    app()
