import typer
from music_transcribe import __version__

app = typer.Typer(help="Letra, cifra e tab de uma música a partir do áudio, localmente.", no_args_is_help=True)


def _version(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version, is_eager=True)):
    """music-transcribe"""
