from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable
import typer
from music_transcribe.models import default_cache_dir

STEM_NAMES = ["vocals", "guitar", "bass", "drums", "piano", "other"]
MODEL = "htdemucs_6s"
# htdemucs_6s is a single-model bag (demucs/remote/htdemucs_6s.yaml: models ['5c90dfd2']);
# torch.hub stores it as hub/checkpoints/5c90dfd2-<hash>.th — the file name has no "6s" in it.
CHECKPOINT_SIG = "5c90dfd2"
CHECKPOINT_MB = 55


def torch_home() -> Path:
    return default_cache_dir() / "torch"


def _system_checkpoint_dirs() -> list[Path]:
    """Where torch.hub keeps checkpoints when TORCH_HOME is not overridden (reused, never written)."""
    dirs = []
    if os.environ.get("TORCH_HOME"):
        dirs.append(Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints")
    if os.environ.get("XDG_CACHE_HOME"):
        dirs.append(Path(os.environ["XDG_CACHE_HOME"]) / "torch" / "hub" / "checkpoints")
    dirs.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
    return dirs


def _find_checkpoint(d: Path) -> Path | None:
    return next(iter(sorted(d.glob(f"{CHECKPOINT_SIG}-*.th"))), None) if d.is_dir() else None


def ensure_demucs_weights(home: Path, confirm: Callable[[str, int], bool],
                          reuse_dirs: list[Path] | None = None) -> None:
    """Make sure the htdemucs_6s checkpoint is under <home>/hub/checkpoints: already there, or linked
    from a torch.hub cache that has it, or (after confirm) left for demucs to download on first run."""
    ck = Path(home) / "hub" / "checkpoints"
    if _find_checkpoint(ck):
        return
    for d in (_system_checkpoint_dirs() if reuse_dirs is None else reuse_dirs):
        f = _find_checkpoint(d)
        if f and d.resolve() != ck.resolve():
            ck.mkdir(parents=True, exist_ok=True)
            (ck / f.name).symlink_to(f.resolve())
            return
    if not confirm(MODEL, CHECKPOINT_MB):
        raise RuntimeError(f"pesos do demucs ({MODEL}, ~{CHECKPOINT_MB} MB) não baixados; rode com --yes")


def separate(audio: Path, out_dir: Path, runner=subprocess.run, device: str = "mps",
             confirm: Callable[[str, int], bool] | None = None) -> dict[str, Path]:
    audio = Path(audio); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    home = torch_home()
    ensure_demucs_weights(home, confirm or (lambda name, mb: True))
    env = {**os.environ, "TORCH_HOME": str(home)}
    work = out_dir / "_demucs"

    def attempt(dev: str):
        shutil.rmtree(work, ignore_errors=True)
        cmd = [sys.executable, "-m", "demucs", "-n", MODEL, "-d", dev, "-o", str(work), str(audio)]
        return runner(cmd, check=False, capture_output=True, text=True, env=env)

    r = attempt(device)
    if getattr(r, "returncode", 0) != 0 and device != "cpu":
        typer.echo(f"demucs falhou em {device}; tentando de novo em cpu", err=True)
        r = attempt("cpu")
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
