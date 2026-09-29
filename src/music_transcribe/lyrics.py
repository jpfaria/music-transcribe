from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path
from music_transcribe.schema import LyricLine

NON_SPEECH = {"[MUSIC]", "[music]", "♪", "♪♪", "(music)", "[BLANK_AUDIO]", "(instrumental)"}


def parse_whisper_json(data: dict) -> list[LyricLine]:
    out: list[LyricLine] = []
    prev_text: str | None = None
    prev_conf: float = 1.0
    for seg in data.get("transcription", []):
        text = seg.get("text", "").strip().strip("♪").strip()
        if not text or text in NON_SPEECH:
            continue
        if re.search(r"[^\W_]", text) is None:
            continue  # no letter or digit: punctuation-only hallucination (e.g. ". . .")
        toks = [t for t in seg.get("tokens", []) if not t.get("text", "").startswith("[_")]
        ps = [float(t["p"]) for t in toks if "p" in t]
        conf = sum(ps) / len(ps) if ps else 0.5
        if text == prev_text and conf < 0.6 and prev_conf < 0.6:
            continue  # repeated low-confidence hallucination
        off = seg.get("offsets", {})
        out.append(LyricLine(off.get("from", 0) / 1000.0, off.get("to", 0) / 1000.0, text, round(conf, 3)))
        prev_text, prev_conf = text, conf
    return out


def transcribe(vocals_wav: Path, model_path: Path, runner=subprocess.run, prompt: str = "Song lyrics:",
               language: str = "en") -> list[LyricLine]:
    """vocals_wav must be 16 kHz mono (use audio.to_wav(sr=16000, mono=True))."""
    of = str(Path(vocals_wav).with_suffix(""))
    cmd = ["whisper-cli", "-m", str(model_path), "-f", str(vocals_wav), "-l", language,
           "-sns", "-np", "-bs", "8", "--prompt", prompt, "-oj", "-of", of]
    r = runner(cmd, check=False, capture_output=True, text=True)
    if getattr(r, "returncode", 0) != 0:
        raise RuntimeError(f"whisper-cli failed: {getattr(r, 'stderr', '')[-2000:]}")
    data = json.loads(Path(of + ".json").read_text())
    return parse_whisper_json(data)
