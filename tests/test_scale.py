from music_transcribe.render.scale import scale_for


def test_minor_key_minor_pentatonic():
    s = scale_for("Bbm")
    assert s["notes"] == ["Bb", "Db", "Eb", "F", "Ab"]
    assert s["box_fret"] == 6
    assert "menor pentatônica" in s["name"]


def test_major_key_major_pentatonic_is_relative_minor():
    s = scale_for("C")
    assert s["notes"] == ["C", "D", "E", "G", "A"]
    assert "Am" in s["name"] and s["box_fret"] == 8


def test_e_minor_box_is_open_position():
    assert scale_for("Em")["box_fret"] == 0
