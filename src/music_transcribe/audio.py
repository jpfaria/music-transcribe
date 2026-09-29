from __future__ import annotations
import subprocess
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf


def load_mono(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), sr


def duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def to_wav(src: Path, dst: Path, sr: int | None = None, mono: bool = False, runner=subprocess.run) -> Path:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn"]
    if sr:
        cmd += ["-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd.append(str(dst))
    r = runner(cmd, check=False, capture_output=True, text=True)
    if getattr(r, "returncode", 0) != 0:
        raise RuntimeError(f"ffmpeg failed: {getattr(r, 'stderr', '')}")
    return Path(dst)
