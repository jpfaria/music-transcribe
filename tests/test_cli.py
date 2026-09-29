from typer.testing import CliRunner
from music_transcribe.cli import app
from music_transcribe import __version__

runner = CliRunner()


def test_version():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0
    assert __version__ in r.output
