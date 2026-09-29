import json
from music_transcribe.tags import read_tags

FFPROBE = {"format": {"tags": {"TITLE": "Midnight on My Mind", "ARTIST": "Chicago Blues Radio", "ALBUM": "Vol. 007", "LYRICS": "la la"}}}


def test_read_tags_parses_ffprobe():
    def runner(cmd, **kw):
        class R: returncode = 0; stdout = json.dumps(FFPROBE)
        return R()
    t = read_tags("x.flac", runner=runner)
    assert t.title == "Midnight on My Mind"
    assert t.artist == "Chicago Blues Radio"
    assert t.embedded_lyrics == "la la"


def test_read_tags_missing_is_empty():
    def runner(cmd, **kw):
        class R: returncode = 0; stdout = json.dumps({"format": {}})
        return R()
    t = read_tags("x.flac", runner=runner)
    assert t.title == "" and t.embedded_lyrics == ""
