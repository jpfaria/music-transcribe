from __future__ import annotations
from pathlib import Path
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


AUDIO = typer.Argument(..., exists=True, dir_okay=False)
OUT = typer.Option(None, "--out", help="Pasta de saída (padrão: <audio>.transcribe/)")
FORCE = typer.Option(False, "--force", help="Refaz a etapa mesmo com cache")


@app.command()
def run(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, yes: bool = typer.Option(False, "--yes"),
        model: str = typer.Option("large-v3", "--model"), instruments: str = typer.Option("guitar,bass,piano", "--instruments"),
        device: str = typer.Option("mps", "--device")):
    """Roda o pipeline inteiro: tags → stems → lyrics → harmony → notes → render."""
    o = _out(audio, out)
    pipeline.stage_tags(audio, o, force)
    pipeline.stage_stems(audio, o, force, device)
    pipeline.stage_lyrics(audio, o, model, _confirm(yes), force)
    pipeline.stage_harmony(audio, o, force)
    pipeline.stage_notes(audio, o, [i.strip() for i in instruments.split(",") if i.strip()], force)
    files = pipeline.stage_render(o)
    for k, p in files.items():
        typer.echo(f"{k}: {p}")


@app.command()
def tags(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE):
    pipeline.stage_tags(audio, _out(audio, out), force)


@app.command()
def stems(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, device: str = typer.Option("mps", "--device")):
    for k, p in pipeline.stage_stems(audio, _out(audio, out), force, device).items():
        typer.echo(f"{k}: {p}")


@app.command()
def lyrics(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, yes: bool = typer.Option(False, "--yes"),
           model: str = typer.Option("large-v3", "--model")):
    pipeline.stage_lyrics(audio, _out(audio, out), model, _confirm(yes), force)


@app.command()
def harmony(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE):
    h = pipeline.stage_harmony(audio, _out(audio, out), force)
    typer.echo(f"{h.key} · {h.bpm} BPM · {h.meter} · loop {h.loop}")


@app.command()
def notes(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE,
          instruments: str = typer.Option("guitar,bass,piano", "--instruments")):
    pipeline.stage_notes(audio, _out(audio, out), [i.strip() for i in instruments.split(",")], force)


@app.command()
def render(audio: Path = AUDIO, out: Path | None = OUT):
    for k, p in pipeline.stage_render(_out(audio, out)).items():
        typer.echo(f"{k}: {p}")


if __name__ == "__main__":
    app()
