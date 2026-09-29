import pretty_midi
from music_transcribe.render.midi import write_midi
from music_transcribe.schema import Note


def test_write_midi(tmp_path):
    p = tmp_path / "g.mid"
    write_midi([Note(0.0, 0.5, 58, 0.9), Note(0.5, 1.0, 61, 0.4)], p)
    pm = pretty_midi.PrettyMIDI(str(p))
    assert [n.pitch for n in pm.instruments[0].notes] == [58, 61]
