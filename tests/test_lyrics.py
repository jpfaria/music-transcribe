import numpy as np
from music_transcribe.lyrics import parse_whisper_json, transcribe, vocal_activity, filter_by_activity
from music_transcribe.schema import LyricLine

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


def _seg(text, p=0.4, from_ms=0, to_ms=1000):
    return {"offsets": {"from": from_ms, "to": to_ms}, "text": text,
            "tokens": [{"text": t, "p": p} for t in text.split()] or [{"text": text, "p": p}]}


def test_parse_drops_punctuation_only_hallucinations():
    for text in [". . .", "...", "♪ ♪"]:
        assert parse_whisper_json({"transcription": [_seg(text)]}) == []


def test_parse_drops_repeated_low_confidence_hallucination():
    data = {"transcription": [_seg("la la", p=0.4, from_ms=0, to_ms=1000),
                               _seg("la la", p=0.4, from_ms=1000, to_ms=2000)]}
    lines = parse_whisper_json(data)
    assert len(lines) == 1
    assert lines[0].text == "la la"


def test_parse_keeps_normal_line_unaffected():
    lines = parse_whisper_json(WJ)
    assert len(lines) == 1 and lines[0].text == "Moonlight spills on a crooked street"


def test_transcribe_builds_command_and_reads_json(tmp_path):
    import json
    wav = tmp_path / "vocals16.wav"; wav.write_bytes(b"RIFF")
    def runner(cmd, **kw):
        assert cmd[0] == "whisper-cli" and "-ojf" in cmd and "-sns" in cmd and "--prompt" in cmd
        open(cmd[cmd.index("-of") + 1] + ".json", "w").write(json.dumps(WJ))
        class R: returncode = 0; stderr = ""
        return R()
    lines = transcribe(wav, tmp_path / "ggml-large-v3.bin", runner=runner)
    assert len(lines) == 1


def test_transcribe_tolerates_invalid_utf8_in_ojf_token_dump(tmp_path):
    # -ojf can dump a raw fragment of a multi-byte UTF-8 character as a token's own
    # "text" (whisper.cpp's per-token text isn't reassembled the way the segment text is).
    wav = tmp_path / "vocals16.wav"; wav.write_bytes(b"RIFF")

    def runner(cmd, **kw):
        assert "-ojf" in cmd
        raw = (b'{"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": " hi",'
               b' "tokens": [{"text": " h", "p": 0.9}, {"text": "i \xe2\x99", "p": 0.8}]}]}')
        with open(cmd[cmd.index("-of") + 1] + ".json", "wb") as f:
            f.write(raw)
        class R: returncode = 0; stderr = ""
        return R()
    lines = transcribe(wav, tmp_path / "ggml-large-v3.bin", runner=runner)
    assert len(lines) == 1 and lines[0].text == "hi"


def test_vocal_activity_finds_sine_burst_in_silence():
    sr = 16000
    y = np.zeros(3 * sr, dtype=np.float32)
    t = np.arange(sr) / sr
    y[sr:2 * sr] = 0.5 * np.sin(2 * np.pi * 220 * t)
    regions = vocal_activity(y, sr)
    assert len(regions) == 1
    start, end = regions[0]
    assert abs(start - 1.0) < 0.15
    assert abs(end - 2.0) < 0.15


def test_vocal_activity_on_silence_is_empty():
    y = np.zeros(3 * 16000, dtype=np.float32)
    assert vocal_activity(y, 16000) == []


def test_filter_by_activity_drops_lines_outside_active_regions():
    lines = [LyricLine(1.0, 2.0, "kept", 0.9), LyricLine(10.0, 12.0, "dropped", 0.9)]
    out = filter_by_activity(lines, [(0.0, 3.0)])
    assert [l.text for l in out] == ["kept"]


def test_vocal_activity_noise_below_absolute_floor_is_empty():
    y = (0.001 * np.random.default_rng(0).standard_normal(5 * 16000)).astype(np.float32)   # RMS −60 dBFS
    assert vocal_activity(y, 16000) == []


def test_vocal_activity_sine_above_floor_is_one_region():
    sr = 16000
    y = np.zeros(3 * sr, dtype=np.float32)
    t = np.arange(sr) / sr
    y[sr:2 * sr] = (0.1 * np.sqrt(2)) * np.sin(2 * np.pi * 220 * t)                     # RMS −20 dBFS
    regions = vocal_activity(y, sr)
    assert len(regions) == 1 and abs(regions[0][0] - 1.0) < 0.15 and abs(regions[0][1] - 2.0) < 0.15
