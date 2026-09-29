from __future__ import annotations
import hashlib
from pathlib import Path

STAGES = ["tags", "stems", "lyrics", "harmony", "notes", "render"]
INPUT_HASH_FILE = "input.sha1"


class OutDir:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def stage(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _marker(sub: str) -> str:
        return f".done.{sub}" if sub else ".done"

    def done(self, name: str, sub: str = "") -> bool:
        return (self.root / name / self._marker(sub)).exists()

    def mark_done(self, name: str, sub: str = "") -> None:
        (self.stage(name) / self._marker(sub)).write_text("ok")

    def invalidate(self, *stages: str) -> None:
        """Remove every .done marker (including per-part ones like notes/.done.guitar) of the given
        stages; with no arguments, of all stages."""
        for name in stages or STAGES:
            d = self.root / name
            if d.is_dir():
                for m in d.glob(".done*"):
                    m.unlink(missing_ok=True)

    def check_input(self, audio: Path) -> bool:
        """Record the input's hash on first use. Returns False (after invalidating every stage and
        storing the new hash) when the audio differs from the one the cache was built from."""
        h = self.input_hash(audio)
        f = self.root / INPUT_HASH_FILE
        old = f.read_text().strip() if f.exists() else None
        if old == h:
            return True
        if old is not None:
            self.invalidate()
        f.write_text(h)
        return old is None

    @staticmethod
    def input_hash(audio: Path) -> str:
        h = hashlib.sha1()
        with open(audio, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
