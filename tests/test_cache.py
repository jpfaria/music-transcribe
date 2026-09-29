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


def test_per_part_markers(tmp_path: Path):
    o = OutDir(tmp_path / "out")
    o.mark_done("notes", "guitar")
    assert o.done("notes", "guitar") and not o.done("notes", "bass") and not o.done("notes")


def test_invalidate_removes_markers(tmp_path: Path):
    o = OutDir(tmp_path / "out")
    for s in ("tags", "harmony"):
        o.mark_done(s)
    o.mark_done("notes", "guitar")
    o.invalidate("notes")
    assert not o.done("notes", "guitar") and o.done("harmony")
    o.invalidate()
    assert not o.done("tags") and not o.done("harmony")


def test_check_input_invalidates_on_changed_audio(tmp_path: Path):
    a = tmp_path / "a.flac"; a.write_bytes(b"abc")
    o = OutDir(tmp_path / "out")
    assert o.check_input(a) is True                     # first run: hash recorded
    assert (o.root / "input.sha1").read_text() == o.input_hash(a)
    o.mark_done("tags"); o.mark_done("notes", "guitar")
    assert o.check_input(a) is True and o.done("tags")  # same audio: cache kept
    a.write_bytes(b"abd")
    assert o.check_input(a) is False
    assert not o.done("tags") and not o.done("notes", "guitar")
    assert (o.root / "input.sha1").read_text() == o.input_hash(a)
