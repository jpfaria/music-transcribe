from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

STEM_NAMES = ["vocals", "guitar", "bass", "drums", "piano", "other"]
MODEL = "htdemucs_6s"


def separate(audio: Path, out_dir: Path, runner=subprocess.run, device: str = "mps") -> dict[str, Path]:
    audio = Path(audio); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_demucs"
    cmd = [sys.executable, "-m", "demucs", "-n", MODEL, "-d", device, "-o", str(work), str(audio)]
    r = runner(cmd, check=False, capture_output=True, text=True)
    if getattr(r, "returncode", 0) != 0:
        raise RuntimeError(f"demucs failed: {getattr(r, 'stderr', '')[-2000:]}")
    src = work / MODEL / audio.stem
    stems = {}
    for name in STEM_NAMES:
        f = src / f"{name}.wav"
        if not f.exists():
            raise RuntimeError(f"demucs did not produce stem '{name}' in {src}")
        dst = out_dir / f"{name}.wav"
        shutil.move(str(f), str(dst))
        stems[name] = dst
    shutil.rmtree(work, ignore_errors=True)
    return stems
