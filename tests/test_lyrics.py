from music_transcribe.lyrics import parse_whisper_json, transcribe

WJ = {"transcription": [
    {"timestamps": {"from": "00:00:20,400", "to": "00:00:25,200"}, "offsets": {"from": 20400, "to": 25200},
     "text": " Moonlight spills on a crooked street", "tokens": [{"text": "Moon", "p": 0.9}, {"text": "light", "p": 0.7}]},
    {"timestamps": {"from": "00:01:00,000", "to": "00:01:10,000"}, "offsets": {"from": 60000, "to": 70000},
     "text": " [MUSIC]", "tokens": [{"text": "[MUSIC]", "p": 0.99}]},
]}


def test_parse_drops_non_speech_and_averages_confidence():
    lines = parse_whisper_json(WJ)
    assert len(lines) == 1
    l = lines[0]
    assert l.text == "Moonlight spills on a crooked street"
    assert abs(l.start - 20.4) < 1e-6 and abs(l.end - 25.2) < 1e-6
    assert abs(l.confidence - 0.8) < 1e-6


def test_parse_empty_is_instrumental():
    assert parse_whisper_json({"transcription": []}) == []


def test_transcribe_builds_command_and_reads_json(tmp_path):
    import json
    wav = tmp_path / "vocals16.wav"; wav.write_bytes(b"RIFF")
    def runner(cmd, **kw):
        assert cmd[0] == "whisper-cli" and "-oj" in cmd and "-sns" in cmd and "--prompt" in cmd
        open(cmd[cmd.index("-of") + 1] + ".json", "w").write(json.dumps(WJ))
        class R: returncode = 0; stderr = ""
        return R()
    lines = transcribe(wav, tmp_path / "ggml-large-v3.bin", runner=runner)
    assert len(lines) == 1
