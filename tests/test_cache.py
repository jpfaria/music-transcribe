from pathlib import Path
from music_transcribe.cache import OutDir


def test_stage_and_done(tmp_path: Path):
    o = OutDir(tmp_path / "out")
    s = o.stage("lyrics")
    assert s.is_dir()
    assert not o.done("lyrics")
    o.mark_done("lyrics")
    assert o.done("lyrics")


def test_input_hash_changes_with_content(tmp_path: Path):
    a = tmp_path / "a.bin"; a.write_bytes(b"abc")
    b = tmp_path / "b.bin"; b.write_bytes(b"abd")
    o = OutDir(tmp_path / "out")
    assert o.input_hash(a) != o.input_hash(b)
    assert o.input_hash(a) == o.input_hash(a)
