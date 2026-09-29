from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

_HF = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
WHISPER_MODELS: dict[str, tuple[str, int]] = {
    "large-v3": (_HF + "ggml-large-v3.bin", 3095),
    "medium.en": (_HF + "ggml-medium.en.bin", 1533),
    "medium": (_HF + "ggml-medium.bin", 1533),
    "small.en": (_HF + "ggml-small.en.bin", 488),
}


def default_cache_dir() -> Path:
    return Path(os.environ.get("MUSIC_TRANSCRIBE_CACHE", Path.home() / ".cache" / "music-transcribe"))


# whisper.cpp's own download script keeps models here; reused as-is, never written to.
WHISPER_CPP_DIR = Path.home() / ".cache" / "whisper"


def _curl(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(".part")
    try:
        subprocess.run(["curl", "-L", "--fail", "--progress-bar", "-o", str(tmp), url], check=True)
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download falhou: {url} (curl saiu com {e.returncode})") from e
    tmp.rename(dst)


def _usable(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def ensure_whisper_model(
    name: str,
    confirm: Callable[[str, int], bool],
    cache_dir: Path | None = None,
    downloader: Callable[[str, Path], None] | None = None,
    reuse_dir: Path | None = None,
) -> Path | None:
    if name not in WHISPER_MODELS:
        raise ValueError(f"modelo Whisper desconhecido: {name!r} (use {', '.join(WHISPER_MODELS)})")
    url, size_mb = WHISPER_MODELS[name]
    cache = Path(cache_dir or default_cache_dir())
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"ggml-{name}.bin"
    if _usable(dst):
        return dst
    shared = Path(reuse_dir or WHISPER_CPP_DIR) / f"ggml-{name}.bin"
    if _usable(shared):
        return shared
    if not confirm(name, size_mb):
        return None
    (downloader or _curl)(url, dst)
    return dst if dst.exists() else None
