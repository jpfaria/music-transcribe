from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints, get_origin, get_args

NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def pc_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


def pitch_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    amplitude: float
    articulation: str = ""      # "", "bend:1", "bend:2", "bend-release", "vibrato", "slide:<to_pitch>", "hammer", "pull"
    confidence: str = "high"    # high | medium | low
    reason: str = ""            # why confidence < high, e.g. "bleed", "weak", "unstable"
    string: int | None = None   # 0 = highest string
    fret: int | None = None


@dataclass
class Chord:
    bar: int
    start: float
    name: str        # "Bbm", "Db/F", "Ebm7b5"
    root: str
    quality: str     # key of chords.TEMPLATES
    bass: str
    confidence: float  # margin best - second, 0..1


@dataclass
class Harmony:
    bpm: float
    meter: str       # "4/4" | "12/8"
    key: str         # "Bbm" | "Db"
    bar0: float
    bar_len: float
    chords: list[Chord] = field(default_factory=list)
    loop: list[str] = field(default_factory=list)


@dataclass
class LyricLine:
    start: float
    end: float
    text: str
    confidence: float


@dataclass
class Tags:
    title: str = ""
    artist: str = ""
    album: str = ""
    embedded_lyrics: str = ""


T = TypeVar("T")


def _from_dict(cls: type[T], d: dict[str, Any]) -> T:
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        v = d.get(f.name)
        t = hints[f.name]
        if get_origin(t) is list and v is not None:
            (inner,) = get_args(t)
            if is_dataclass(inner):
                v = [_from_dict(inner, x) for x in v]
        elif is_dataclass(t) and isinstance(v, dict):
            v = _from_dict(t, v)
        kwargs[f.name] = v
    return cls(**kwargs)


def save_json(obj: Any, path: Path) -> None:
    data = [asdict(o) for o in obj] if isinstance(obj, list) else asdict(obj)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_json(cls: type[T], path: Path) -> T | list[T]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [_from_dict(cls, x) for x in data]
    return _from_dict(cls, data)
