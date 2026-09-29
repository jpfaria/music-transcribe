from __future__ import annotations
import hashlib
from pathlib import Path

STAGES = ["tags", "stems", "lyrics", "harmony", "notes", "render"]


class OutDir:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def stage(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def done(self, name: str) -> bool:
        return (self.root / name / ".done").exists()

    def mark_done(self, name: str) -> None:
        (self.stage(name) / ".done").write_text("ok")

    @staticmethod
    def input_hash(audio: Path) -> str:
        h = hashlib.sha1()
        with open(audio, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
