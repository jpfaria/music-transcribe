import json, shutil, subprocess, sys
import pytest
from pathlib import Path
from make_fixture import make, ROOTS, LEAD

pytestmark = pytest.mark.slow


@pytest.mark.skipif(shutil.which("whisper-cli") is None or shutil.which("ffmpeg") is None, reason="needs whisper-cli and ffmpeg")
def test_full_pipeline_on_synthetic_song(tmp_path):
    audio = make(tmp_path / "synth.wav")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "music_transcribe.cli", "run", str(audio), "--out", str(out), "--yes",
                        "--model", "small.en", "--instruments", "guitar,bass,other", "--device", "cpu", "--language", "en"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    h = json.loads((out / "harmony" / "harmony.json").read_text())
    assert 55 <= h["bpm"] <= 65
    assert h["key"] in ("Bbm", "Db")
    names = [c["name"] for c in h["chords"]]
    assert names[:4] == ["Bbm", "Db", "Ebm", "Bbm"] or h["loop"] == ["Bbm", "Db", "Ebm", "Bbm"]
    assert (out / "render" / "cifra.html").exists()
    assert json.loads((out / "lyrics" / "lyrics.json").read_text()) == []   # instrumental
    bass = json.loads((out / "notes" / "bass.json").read_text())
    assert {n["pitch"] % 12 for n in bass} == {r % 12 for r in ROOTS}
    # demucs routes the pure-sine lead partly to "other" (bar 8 is ~-86 dB in guitar, -19 dB in other),
    # so the lead is checked across the two stems a lead guitar can land in.
    lead_stems = [json.loads((out / "notes" / f"{i}.json").read_text()) for i in ("guitar", "other")]
    assert set(LEAD) <= {n["pitch"] for notes in lead_stems for n in notes}
