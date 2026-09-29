from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path
import librosa
import numpy as np
from music_transcribe.audio import load_mono
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
           "-sns", "-np", "-bs", "8", "--prompt", prompt, "-ojf", "-of", of]
    r = runner(cmd, check=False, capture_output=True, text=True)
    if getattr(r, "returncode", 0) != 0:
        raise RuntimeError(f"whisper-cli failed: {getattr(r, 'stderr', '')[-2000:]}")
    # -ojf dumps individual sub-word tokens, which can be a raw fragment of a multi-byte
    # UTF-8 character (e.g. half of "♪") and are not guaranteed valid UTF-8 on their own.
    raw = Path(of + ".json").read_bytes().decode("utf-8", errors="replace")
    data = json.loads(raw)
    return parse_whisper_json(data)


def vocal_activity(y: np.ndarray, sr: int, hop: int = 512, threshold_db: float = -40.0,
                    min_dur: float = 0.3) -> list[tuple[float, float]]:
    """Return (start, end) regions where RMS energy is within threshold_db of the signal's peak RMS."""
    rms = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop)[0]
    if len(rms) == 0:
        return []
    max_rms = float(rms.max())
    if max_rms <= 0.0:
        return []
    db = 20 * np.log10(np.maximum(rms, 1e-12) / max_rms)
    active = db >= threshold_db
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    frame_dur = hop / sr
    regions: list[tuple[float, float]] = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = float(times[i])
        elif not a and start is not None:
            end = float(times[i])
            if end - start >= min_dur:
                regions.append((start, end))
            start = None
    if start is not None:
        end = float(times[-1] + frame_dur)
        if end - start >= min_dur:
            regions.append((start, end))
    return regions


def vocal_activity_for(path: Path, sr: int = 16000, **kw) -> list[tuple[float, float]]:
    y, sr = load_mono(path, sr=sr)
    return vocal_activity(y, sr, **kw)


def filter_by_activity(lines: list[LyricLine], regions: list[tuple[float, float]],
                        min_overlap: float = 0.3) -> list[LyricLine]:
    """Keep only lines whose overlap with the active regions is at least min_overlap of their duration."""
    out = []
    for line in lines:
        dur = max(line.end - line.start, 1e-9)
        covered = sum(max(0.0, min(line.end, r1) - max(line.start, r0)) for r0, r1 in regions)
        if covered / dur >= min_overlap:
            out.append(line)
    return out
