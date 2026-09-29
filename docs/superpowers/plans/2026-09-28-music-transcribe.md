# music-transcribe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local CLI + Claude Code skill that turns an audio file into a complete cifra: lyrics, key, tempo, chords per bar with diagrams, and per-instrument tab with articulations and per-item confidence.

**Architecture:** A staged pipeline (`tags → stems → lyrics → harmony → notes → render`), each stage a pure-ish module that reads the previous stage's JSON from `<out>/` and writes its own, so stages cache and can run alone. Heavy tools (demucs, whisper-cli, basic-pitch) are wrapped behind thin adapters that take an injectable `runner` so tests never touch models. Analysis math (tempo, chords, articulations, fretboard, diagrams) lives in pure functions tested on synthetic signals.

**Tech Stack:** Python 3.11, `uv`, typer, numpy, scipy, librosa, soundfile, pretty_midi, basic-pitch (CoreML backend), demucs + torch (MPS), `whisper-cli` (whisper.cpp, Homebrew), ffmpeg/ffprobe.

**Spec:** `docs/superpowers/specs/2026-09-28-music-transcribe-design.md`

## Global Constraints

- Python `>=3.11,<3.12` (basic-pitch has no 3.12 wheel). `uv` manages the venv; `uv run pytest` is the test command.
- Pin `setuptools<70` in dependencies: `resampy` (pulled by basic-pitch) imports `pkg_resources`, removed in newer setuptools.
- No audio is ever sent off-machine. Models download only after `confirm(name, size_mb)` returns True; default cache dir `~/.cache/music-transcribe/`.
- No commercial audio in the repo. Fixtures are synthesized with numpy. `.gitignore` already blocks `*.flac *.wav *.mp3 *.m4a`.
- Every note, chord and lyric line carries a confidence; render never hides low confidence.
- Repo layout mirrors `jpfaria/mackie-control`: `.claude-plugin/plugin.json` + `marketplace.json`, `skills/<name>/SKILL.md`, `src/`, `tests/`, `docs/`, `CLAUDE.md`.
- Commit after every task; push after every commit (`git push`).
- Note names use flats: `C Db D Eb E F Gb G Ab A Bb B`.

## Review Focus

1. **Mono/odd sample rates**: a 48 kHz mono MP3 must flow through every stage; `audio.load_mono` resamples to 22050 and the stems adapter converts input to WAV first. Test in Task 3.
2. **Silence / no vocals** (instrumental track): `lyrics` must return an empty list, not crash, and render must show "instrumental". Test in Task 6 and Task 14.
3. **Tempo octave errors**: a 52 BPM 12/8 shuffle is reported by librosa as 161; the bar-length-vs-bass-change tie-breaker must pick 52. Test in Task 7.
4. **Bass bleed into the guitar stem**: long runs of the same low pitch repeated in triplets must be tagged `low` confidence with reason `bleed`. Test in Task 9.
5. **Chord names outside the shape bank** (e.g. `Ebm7b5`, `Db/F`): the diagram fallback must still produce a valid voicing or an explicit "sem diagrama" placeholder, never an exception. Test in Task 12.

---

## File Structure

```
pyproject.toml
.claude-plugin/plugin.json
.claude-plugin/marketplace.json
CLAUDE.md
README.md
src/music_transcribe/
  __init__.py            version
  cli.py                 typer app: run + one subcommand per stage
  cache.py               OutDir: stage paths, done markers, input hash
  schema.py              dataclasses + json load/save: Note, Chord, Harmony, LyricLine, Tags
  audio.py               load_mono, to_wav (ffmpeg), duration
  models.py              ensure_whisper_model(name, confirm) -> Path
  tags.py                read_tags(path) -> Tags  (ffprobe)
  stems.py               separate(path, out, runner) -> dict[str, Path]  (demucs)
  lyrics.py              transcribe(vocals_wav, model, runner) -> list[LyricLine]
  harmony/__init__.py    analyze(stems) -> Harmony
  harmony/bassroot.py    bass_roots(y, sr) -> list[RootRun]
  harmony/tempo.py       estimate_tempo(y, sr, root_changes) -> (bpm, meter)
  harmony/grid.py        fit_grid(root_changes, bar_len) -> bar0
  harmony/chords.py      TEMPLATES, chord_for_bar(chroma, bass_pc) -> Chord, detect_loop(names)
  notes/__init__.py      transcribe_instrument(wav, instrument, harmony) -> list[Note]
  notes/detect.py        raw_notes(wav) -> list[Note]  (basic-pitch), f0_contour(y, sr, note)
  notes/articulation.py  classify_contour(cents, hop_s), detect_legato(prev, cur, f0_between, onset_ratio)
  notes/confidence.py    score(note, contour_voiced_frac, repeated_low_run) -> ("high"|"medium"|"low", reason)
  notes/fretboard.py     TUNINGS, assign_positions(notes, tuning, preferred_region)
  render/__init__.py     render_all(out) -> None
  render/chord_diagram.py voicing_for(chord_name) -> Voicing | None, svg(voicing)
  render/tab.py          bar_tab(notes, bar_start, bar_len, meter, n_strings) -> str
  render/text.py         cifra_txt(...)
  render/midi.py         write_midi(notes, path)
  render/html.py         cifra_html(...)
skills/transcribing-a-song/SKILL.md
tests/
  conftest.py            synth helpers: sine_note, click_track, chord_pad
  test_schema.py test_cache.py test_audio.py test_tags.py test_models.py test_stems.py
  test_lyrics.py test_bassroot.py test_tempo.py test_grid.py test_chords.py
  test_detect.py test_articulation.py test_confidence.py test_fretboard.py
  test_chord_diagram.py test_tab.py test_text.py test_midi.py test_html.py test_cli.py
  test_integration.py    marked slow
```

---

### Task 1: Scaffold package, CLI skeleton, plugin manifests

**Files:**
- Create: `pyproject.toml`, `src/music_transcribe/__init__.py`, `src/music_transcribe/cli.py`, `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `CLAUDE.md`, `tests/test_cli.py`

**Interfaces:**
- Produces: `music_transcribe.__version__`, typer app `music_transcribe.cli:app`, console script `music-transcribe`.

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "music-transcribe"
version = "0.1.0"
description = "Letra, cifra e tab de uma música a partir do áudio, localmente."
requires-python = ">=3.11,<3.12"
dependencies = [
  "typer>=0.12",
  "numpy>=1.26,<2.0",
  "scipy>=1.11",
  "librosa>=0.10.2",
  "soundfile>=0.12",
  "pretty_midi>=0.2.10",
  "basic-pitch>=0.4.0",
  "demucs>=4.0.1",
  "torch>=2.2",
  "setuptools<70",
]

[project.scripts]
music-transcribe = "music_transcribe.cli:app"

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-timeout>=2.3"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/music_transcribe"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: full pipeline with real models; run manually"]
addopts = "-m 'not slow'"
```

- [ ] **Step 2: Write package init and CLI skeleton**

`src/music_transcribe/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/music_transcribe/cli.py`:
```python
import typer
from music_transcribe import __version__

app = typer.Typer(help="Letra, cifra e tab de uma música a partir do áudio, localmente.", no_args_is_help=True)


def _version(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version, is_eager=True)):
    """music-transcribe"""
```

- [ ] **Step 3: Write the failing test**

`tests/test_cli.py`:
```python
from typer.testing import CliRunner
from music_transcribe.cli import app
from music_transcribe import __version__

runner = CliRunner()


def test_version():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0
    assert __version__ in r.output
```

- [ ] **Step 4: Install and run**

Run: `uv venv -p 3.11 && uv pip install -e ".[dev]" && uv run pytest tests/test_cli.py -v`
Expected: PASS (torch download is large; this is the one-time cost).

- [ ] **Step 5: Plugin manifests and CLAUDE.md**

`.claude-plugin/plugin.json`:
```json
{
  "name": "music-transcribe",
  "displayName": "Music Transcribe",
  "description": "Letra, cifra com diagramas e tab com articulações de uma música a partir do áudio, tudo local.",
  "version": "0.1.0",
  "author": { "name": "João Paulo Faria", "email": "jpfaria@gmail.com" },
  "keywords": ["music", "transcription", "chords", "tab", "lyrics", "whisper", "demucs"],
  "homepage": "https://github.com/jpfaria/music-transcribe",
  "repository": "https://github.com/jpfaria/music-transcribe.git",
  "license": "MIT"
}
```

`.claude-plugin/marketplace.json`:
```json
{
  "name": "music-transcribe",
  "description": "Letra, cifra e tab de uma música a partir do áudio.",
  "owner": { "name": "João Paulo Faria", "email": "jpfaria@gmail.com" },
  "plugins": [
    {
      "name": "music-transcribe",
      "source": { "source": "github", "repo": "jpfaria/music-transcribe" },
      "description": "Letra, cifra com diagramas e tab com articulações de uma música a partir do áudio, tudo local."
    }
  ]
}
```

`CLAUDE.md`:
```markdown
# music-transcribe

CLI Python + skill Claude Code. Pipeline em etapas cacheadas em `<out>/`: tags → stems → lyrics → harmony → notes → render.

- Python 3.11 obrigatório (basic-pitch). `uv run pytest` roda os testes rápidos; `uv run pytest -m slow` roda o pipeline real.
- Ferramentas pesadas ficam atrás de adaptadores com `runner` injetável. Teste de unidade nunca chama demucs, whisper-cli ou basic-pitch.
- Áudio comercial nunca entra no repo. Fixtures são sintetizadas em `tests/conftest.py`.
- Confiança (high/medium/low) acompanha cada nota, acorde e linha de letra até o HTML. Nunca esconder.
- Spec: `docs/superpowers/specs/2026-09-28-music-transcribe-design.md`. Plano: `docs/superpowers/plans/2026-09-28-music-transcribe.md`.
- Aprendizado de projeto vai em `docs/` ou na skill, não em memória de usuário.
```

- [ ] **Step 6: Commit and push**

```bash
git add -A && git commit -m "Scaffold package, CLI skeleton and plugin manifests" && git push
```

---

### Task 2: Schema and output cache

**Files:**
- Create: `src/music_transcribe/schema.py`, `src/music_transcribe/cache.py`, `tests/test_schema.py`, `tests/test_cache.py`

**Interfaces:**
- Produces:
  - `NOTE_NAMES: list[str]`, `pitch_name(midi:int)->str`, `pc_name(pc:int)->str`
  - `@dataclass Note(start, end, pitch, amplitude, articulation="", confidence="high", reason="", string=None, fret=None)`
  - `@dataclass Chord(bar:int, start:float, name:str, root:str, quality:str, bass:str, confidence:float)`
  - `@dataclass Harmony(bpm:float, meter:str, key:str, bar0:float, bar_len:float, chords:list[Chord], loop:list[str])`
  - `@dataclass LyricLine(start, end, text, confidence)`
  - `@dataclass Tags(title, artist, album, embedded_lyrics)`
  - `save_json(obj_or_list, path)`, `load_json(cls, path)` (handles list or single)
  - `OutDir(root: Path)` with `.stage(name)->Path` (mkdir), `.done(name)->bool`, `.mark_done(name)`, `.input_hash(audio)->str`

- [ ] **Step 1: Failing tests**

`tests/test_schema.py`:
```python
from pathlib import Path
from music_transcribe.schema import Note, Chord, Harmony, LyricLine, pitch_name, save_json, load_json


def test_pitch_name():
    assert pitch_name(58) == "Bb3"
    assert pitch_name(60) == "C4"


def test_roundtrip_list(tmp_path: Path):
    notes = [Note(0.0, 0.5, 58, 0.9, articulation="bend:2", confidence="low", reason="bleed", string=1, fret=6)]
    p = tmp_path / "n.json"
    save_json(notes, p)
    back = load_json(Note, p)
    assert back == notes


def test_roundtrip_nested(tmp_path: Path):
    h = Harmony(52.0, "12/8", "Bbm", 0.75, 4.64, [Chord(1, 0.75, "Bbm", "Bb", "min", "Bb", 0.3)], ["Bbm", "Db", "Ebm", "Bbm"])
    p = tmp_path / "h.json"
    save_json(h, p)
    assert load_json(Harmony, p) == h
```

`tests/test_cache.py`:
```python
from pathlib import Path
from music_transcribe.cache import OutDir


def test_stage_and_done(tmp_path: Path):
    o = OutDir(tmp_path / "out")
    s = o.stage("lyrics")
    assert s.is_dir()
    assert not o.done("lyrics")
    o.mark_done("lyrics")
    assert o.done("lyrics")


def test_input_hash_changes_with_content(tmp_path: Path):
    a = tmp_path / "a.bin"; a.write_bytes(b"abc")
    b = tmp_path / "b.bin"; b.write_bytes(b"abd")
    o = OutDir(tmp_path / "out")
    assert o.input_hash(a) != o.input_hash(b)
    assert o.input_hash(a) == o.input_hash(a)
```

- [ ] **Step 2: Run, expect ImportError**

Run: `uv run pytest tests/test_schema.py tests/test_cache.py -v`

- [ ] **Step 3: Implement schema.py**

```python
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
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1))


def load_json(cls: type[T], path: Path) -> T | list[T]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return [_from_dict(cls, x) for x in data]
    return _from_dict(cls, data)
```

- [ ] **Step 4: Implement cache.py**

```python
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
```

- [ ] **Step 5: Run tests, expect PASS; commit and push**

```bash
uv run pytest tests/test_schema.py tests/test_cache.py -v
git add -A && git commit -m "Add schema dataclasses and output cache" && git push
```

---

### Task 3: Audio helpers and tags (ffprobe)

**Files:**
- Create: `src/music_transcribe/audio.py`, `src/music_transcribe/tags.py`, `tests/conftest.py`, `tests/test_audio.py`, `tests/test_tags.py`

**Interfaces:**
- Produces:
  - `audio.load_mono(path, sr=22050) -> tuple[np.ndarray, int]`
  - `audio.to_wav(src, dst, sr=None, mono=False, runner=subprocess.run) -> Path`
  - `audio.duration(path) -> float`
  - `tags.read_tags(path, runner=subprocess.run) -> Tags`
  - conftest fixtures: `sine_note(freq, dur, sr, amp)`, `write_wav(path, y, sr)`, `click_track(bpm, bars, sr, subdiv)`

- [ ] **Step 1: conftest synth helpers**

`tests/conftest.py`:
```python
import numpy as np
import pytest
import soundfile as sf

SR = 22050


def sine_note(freq: float, dur: float, sr: int = SR, amp: float = 0.5, harmonics: int = 4) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    y = sum((amp / (k + 1)) * np.sin(2 * np.pi * freq * (k + 1) * t) for k in range(harmonics))
    env = np.minimum(1.0, t / 0.01) * np.exp(-t * 1.5)
    return (y * env).astype(np.float32)


def midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def click_track(bpm: float, bars: int, sr: int = SR, subdiv: int = 3, beats_per_bar: int = 4) -> np.ndarray:
    beat = 60.0 / bpm
    total = int(bars * beats_per_bar * beat * sr) + sr
    y = np.zeros(total, dtype=np.float32)
    n = 0
    while True:
        for s in range(subdiv):
            t = (n + s / subdiv) * beat
            i = int(t * sr)
            if i >= total - 200:
                return y
            amp = 1.0 if s == 0 else 0.4
            y[i:i + 200] += amp * np.hanning(200) * np.sign(np.random.randn(200))
        n += 1


def write_wav(path, y: np.ndarray, sr: int = SR):
    sf.write(str(path), y, sr)
    return path


@pytest.fixture
def sr():
    return SR
```

- [ ] **Step 2: Failing tests**

`tests/test_audio.py`:
```python
import numpy as np
from music_transcribe.audio import load_mono, to_wav, duration
from conftest import sine_note, write_wav


def test_load_mono_resamples_stereo_48k(tmp_path):
    y = sine_note(440, 1.0, sr=48000)
    stereo = np.stack([y, y], axis=1)
    p = write_wav(tmp_path / "s.wav", stereo, 48000)
    m, sr = load_mono(p, sr=22050)
    assert sr == 22050 and m.ndim == 1
    assert abs(len(m) - 22050) < 50


def test_to_wav_uses_runner(tmp_path):
    calls = []
    def runner(cmd, **kw):
        calls.append(cmd); (tmp_path / "o.wav").write_bytes(b"RIFF")
        class R: returncode = 0
        return R()
    out = to_wav(tmp_path / "in.flac", tmp_path / "o.wav", sr=16000, mono=True, runner=runner)
    assert out.exists()
    cmd = calls[0]
    assert cmd[0] == "ffmpeg" and "-ar" in cmd and "16000" in cmd and "-ac" in cmd


def test_duration(tmp_path):
    p = write_wav(tmp_path / "d.wav", sine_note(220, 2.0), 22050)
    assert abs(duration(p) - 2.0) < 0.01
```

`tests/test_tags.py`:
```python
import json
from music_transcribe.tags import read_tags

FFPROBE = {"format": {"tags": {"TITLE": "Midnight on My Mind", "ARTIST": "Chicago Blues Radio", "ALBUM": "Vol. 007", "LYRICS": "la la"}}}


def test_read_tags_parses_ffprobe():
    def runner(cmd, **kw):
        class R: returncode = 0; stdout = json.dumps(FFPROBE)
        return R()
    t = read_tags("x.flac", runner=runner)
    assert t.title == "Midnight on My Mind"
    assert t.artist == "Chicago Blues Radio"
    assert t.embedded_lyrics == "la la"


def test_read_tags_missing_is_empty():
    def runner(cmd, **kw):
        class R: returncode = 0; stdout = json.dumps({"format": {}})
        return R()
    t = read_tags("x.flac", runner=runner)
    assert t.title == "" and t.embedded_lyrics == ""
```

- [ ] **Step 3: Implement audio.py**

```python
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
```

- [ ] **Step 4: Implement tags.py**

```python
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from music_transcribe.schema import Tags

_KEYS = {"title": ["TITLE", "title"], "artist": ["ARTIST", "artist", "album_artist"], "album": ["ALBUM", "album"],
         "embedded_lyrics": ["LYRICS", "lyrics", "UNSYNCEDLYRICS", "unsyncedlyrics"]}


def read_tags(path: Path, runner=subprocess.run) -> Tags:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]
    r = runner(cmd, check=False, capture_output=True, text=True)
    try:
        tags = json.loads(r.stdout).get("format", {}).get("tags", {}) or {}
    except (json.JSONDecodeError, AttributeError):
        tags = {}
    out = {}
    for field, keys in _KEYS.items():
        out[field] = next((str(tags[k]) for k in keys if k in tags), "")
    return Tags(**out)
```

- [ ] **Step 5: Run, PASS, commit, push**

```bash
uv run pytest tests/test_audio.py tests/test_tags.py -v
git add -A && git commit -m "Add audio helpers and ffprobe tag reader" && git push
```

---

### Task 4: Whisper model download with confirmation

**Files:**
- Create: `src/music_transcribe/models.py`, `tests/test_models.py`

**Interfaces:**
- Produces: `WHISPER_MODELS: dict[str, tuple[str, int]]` (name → (url, size_mb)); `ensure_whisper_model(name, confirm, cache_dir=None, downloader=None) -> Path | None`; `default_cache_dir() -> Path`.

- [ ] **Step 1: Failing tests**

```python
from pathlib import Path
from music_transcribe.models import ensure_whisper_model, WHISPER_MODELS


def test_existing_model_returns_without_confirm(tmp_path):
    p = tmp_path / "ggml-large-v3.bin"; p.write_bytes(b"x" * 10)
    called = []
    out = ensure_whisper_model("large-v3", confirm=lambda n, mb: called.append(1) or False, cache_dir=tmp_path)
    assert out == p and not called


def test_declined_download_returns_none(tmp_path):
    out = ensure_whisper_model("large-v3", confirm=lambda n, mb: False, cache_dir=tmp_path, downloader=lambda u, d: None)
    assert out is None


def test_accepted_download_calls_downloader_with_url(tmp_path):
    got = {}
    def dl(url, dst): got["url"] = url; Path(dst).write_bytes(b"model")
    out = ensure_whisper_model("medium.en", confirm=lambda n, mb: True, cache_dir=tmp_path, downloader=dl)
    assert out and out.exists()
    assert got["url"] == WHISPER_MODELS["medium.en"][0]
```

- [ ] **Step 2: Implement**

```python
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Callable

_HF = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
WHISPER_MODELS: dict[str, tuple[str, int]] = {
    "large-v3": (_HF + "ggml-large-v3.bin", 3095),
    "medium.en": (_HF + "ggml-medium.en.bin", 1533),
    "small.en": (_HF + "ggml-small.en.bin", 488),
}


def default_cache_dir() -> Path:
    return Path(os.environ.get("MUSIC_TRANSCRIBE_CACHE", Path.home() / ".cache" / "music-transcribe"))


def _curl(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(".part")
    subprocess.run(["curl", "-L", "--fail", "--progress-bar", "-o", str(tmp), url], check=True)
    tmp.rename(dst)


def ensure_whisper_model(name: str, confirm: Callable[[str, int], bool], cache_dir: Path | None = None,
                         downloader: Callable[[str, Path], None] | None = None) -> Path | None:
    url, size_mb = WHISPER_MODELS[name]
    cache = Path(cache_dir or default_cache_dir())
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"ggml-{name}.bin"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    if not confirm(name, size_mb):
        return None
    (downloader or _curl)(url, dst)
    return dst if dst.exists() else None
```

- [ ] **Step 3: Run, PASS, commit, push**

```bash
uv run pytest tests/test_models.py -v
git add -A && git commit -m "Add whisper model download with size confirmation" && git push
```

---

### Task 5: Stem separation adapter (demucs)

**Files:**
- Create: `src/music_transcribe/stems.py`, `tests/test_stems.py`

**Interfaces:**
- Produces: `STEM_NAMES = ["vocals","guitar","bass","drums","piano","other"]`; `separate(audio, out_dir, runner=subprocess.run, device="mps") -> dict[str, Path]` returning `out_dir/<stem>.wav` for each stem; raises `RuntimeError` naming the missing stem.

- [ ] **Step 1: Failing test**

```python
from pathlib import Path
from music_transcribe.stems import separate, STEM_NAMES


def test_separate_invokes_demucs_and_moves_stems(tmp_path):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "stems"
    def runner(cmd, **kw):
        assert cmd[:3] == ["python", "-m", "demucs"] or cmd[0] == "demucs"
        assert "htdemucs_6s" in cmd
        # emulate demucs output layout: <o>/htdemucs_6s/<track>/<stem>.wav
        d = Path(cmd[cmd.index("-o") + 1]) / "htdemucs_6s" / "song"
        d.mkdir(parents=True)
        for s in STEM_NAMES:
            (d / f"{s}.wav").write_bytes(b"RIFF")
        class R: returncode = 0; stderr = ""
        return R()
    stems = separate(audio, out, runner=runner)
    assert set(stems) == set(STEM_NAMES)
    assert all(p.exists() and p.parent == out for p in stems.values())


def test_separate_missing_stem_raises(tmp_path):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    def runner(cmd, **kw):
        d = Path(cmd[cmd.index("-o") + 1]) / "htdemucs_6s" / "song"; d.mkdir(parents=True)
        (d / "vocals.wav").write_bytes(b"RIFF")
        class R: returncode = 0; stderr = ""
        return R()
    import pytest
    with pytest.raises(RuntimeError, match="guitar"):
        separate(audio, tmp_path / "o", runner=runner)
```

- [ ] **Step 2: Implement**

```python
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
```

Note: the test's first assertion accepts `[sys.executable, "-m", "demucs"]`; adjust the test to `assert cmd[1:3] == ["-m", "demucs"]`.

- [ ] **Step 3: Run, PASS, commit, push**

```bash
uv run pytest tests/test_stems.py -v
git add -A && git commit -m "Add demucs stem separation adapter" && git push
```

---

### Task 6: Lyrics adapter (whisper-cli JSON, confidence, instrumental)

**Files:**
- Create: `src/music_transcribe/lyrics.py`, `tests/test_lyrics.py`

**Interfaces:**
- Consumes: `audio.to_wav`, `models.ensure_whisper_model`.
- Produces: `transcribe(vocals_wav, model_path, runner=subprocess.run, prompt="Song lyrics:") -> list[LyricLine]`; `parse_whisper_json(data: dict) -> list[LyricLine]`; `NON_SPEECH = {"[MUSIC]", "♪", "♪♪", "(music)"}`.

- [ ] **Step 1: Failing tests**

```python
from music_transcribe.lyrics import parse_whisper_json, transcribe

WJ = {"transcription": [
    {"timestamps": {"from": "00:00:20,400", "to": "00:00:25,200"}, "offsets": {"from": 20400, "to": 25200},
     "text": " Moonlight spills on a crooked street", "tokens": [{"text": "Moon", "p": 0.9}, {"text": "light", "p": 0.7}]},
    {"timestamps": {"from": "00:01:00,000", "to": "00:01:10,000"}, "offsets": {"from": 60000, "to": 70000},
     "text": " [MUSIC]", "tokens": [{"text": "[MUSIC]", "p": 0.99}]},
]}


def test_parse_drops_non_speech_and_averages_confidence():
    lines = parse_whisper_json(WJ)
    assert len(lines) == 1
    l = lines[0]
    assert l.text == "Moonlight spills on a crooked street"
    assert abs(l.start - 20.4) < 1e-6 and abs(l.end - 25.2) < 1e-6
    assert abs(l.confidence - 0.8) < 1e-6


def test_parse_empty_is_instrumental():
    assert parse_whisper_json({"transcription": []}) == []


def test_transcribe_builds_command_and_reads_json(tmp_path):
    import json
    wav = tmp_path / "vocals16.wav"; wav.write_bytes(b"RIFF")
    def runner(cmd, **kw):
        assert cmd[0] == "whisper-cli" and "-oj" in cmd and "-sns" in cmd and "--prompt" in cmd
        of = cmd[cmd.index("-of") + 1]
        (tmp_path / (of.split("/")[-1] + ".json")).write_text(json.dumps(WJ)) if False else open(of + ".json", "w").write(json.dumps(WJ))
        class R: returncode = 0; stderr = ""
        return R()
    lines = transcribe(wav, tmp_path / "ggml-large-v3.bin", runner=runner)
    assert len(lines) == 1
```

- [ ] **Step 2: Implement**

```python
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from music_transcribe.schema import LyricLine

NON_SPEECH = {"[MUSIC]", "[music]", "♪", "♪♪", "(music)", "[BLANK_AUDIO]", "(instrumental)"}


def parse_whisper_json(data: dict) -> list[LyricLine]:
    out: list[LyricLine] = []
    for seg in data.get("transcription", []):
        text = seg.get("text", "").strip().strip("♪").strip()
        if not text or text in NON_SPEECH:
            continue
        toks = [t for t in seg.get("tokens", []) if not t.get("text", "").startswith("[_")]
        ps = [float(t["p"]) for t in toks if "p" in t]
        conf = sum(ps) / len(ps) if ps else 0.5
        off = seg.get("offsets", {})
        out.append(LyricLine(off.get("from", 0) / 1000.0, off.get("to", 0) / 1000.0, text, round(conf, 3)))
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
```

Simplify the test's runner to: `open(cmd[cmd.index("-of") + 1] + ".json", "w").write(json.dumps(WJ))`.

- [ ] **Step 3: Run, PASS, commit, push**

```bash
uv run pytest tests/test_lyrics.py -v
git add -A && git commit -m "Add whisper-cli lyrics adapter with per-line confidence" && git push
```

---

### Task 7: Bass roots and tempo/meter estimation

**Files:**
- Create: `src/music_transcribe/harmony/__init__.py` (empty for now), `src/music_transcribe/harmony/bassroot.py`, `src/music_transcribe/harmony/tempo.py`, `tests/test_bassroot.py`, `tests/test_tempo.py`

**Interfaces:**
- Produces:
  - `@dataclass RootRun(start: float, pc: int, dur: float)`; `bass_roots(y, sr, step=0.25, min_dur=0.75) -> list[RootRun]`; `root_changes(runs) -> list[float]` (start times where pc changes).
  - `estimate_tempo(y, sr, root_changes: list[float]) -> tuple[float, str]` returning `(bpm, meter)` where bpm is the dotted-quarter/quarter pulse such that one bar (4 beats) ≈ median root-change interval, and meter is `"12/8"` when onset autocorrelation favours triplet subdivision else `"4/4"`.

- [ ] **Step 1: Failing tests**

`tests/test_bassroot.py`:
```python
import numpy as np
from music_transcribe.harmony.bassroot import bass_roots, root_changes
from conftest import sine_note, midi_to_hz, SR


def test_bass_roots_follow_sequence():
    # Bb1 (34), Db2 (37), Eb2 (39), Bb1 — 2 s each
    y = np.concatenate([sine_note(midi_to_hz(m), 2.0, SR, harmonics=2) for m in [34, 37, 39, 34]])
    runs = bass_roots(y, SR)
    pcs = [r.pc for r in runs if r.dur >= 1.0]
    assert pcs == [10, 1, 3, 10]
    ch = root_changes(runs)
    assert len(ch) == 4 and abs(ch[1] - 2.0) < 0.3
```

`tests/test_tempo.py`:
```python
import numpy as np
from music_transcribe.harmony.tempo import estimate_tempo
from conftest import click_track, SR


def test_shuffle_52_is_not_reported_as_156():
    y = click_track(52, bars=8, sr=SR, subdiv=3)           # 12/8 feel at 52 BPM
    bar = 4 * 60 / 52
    changes = [0.0 + k * bar for k in range(8)]
    bpm, meter = estimate_tempo(y, SR, changes)
    assert abs(bpm - 52) < 3
    assert meter == "12/8"


def test_straight_120_is_4_4():
    y = click_track(120, bars=8, sr=SR, subdiv=2)
    bar = 4 * 60 / 120
    changes = [k * bar for k in range(8)]
    bpm, meter = estimate_tempo(y, SR, changes)
    assert abs(bpm - 120) < 3
    assert meter == "4/4"
```

- [ ] **Step 2: Implement bassroot.py**

```python
from __future__ import annotations
from dataclasses import dataclass
import librosa
import numpy as np


@dataclass
class RootRun:
    start: float
    pc: int
    dur: float


def bass_roots(y: np.ndarray, sr: int, step: float = 0.25, min_dur: float = 0.75,
               fmin: float = 30.0, fmax: float = 200.0) -> list[RootRun]:
    hop = 1024
    f0, vf, vp = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=4096, hop_length=hop)
    t = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    midi = np.round(12 * np.log2(np.nan_to_num(f0, nan=1.0) / 440.0) + 69).astype(int)
    pc = np.where(vf, midi % 12, -1)
    n_bins = int(t[-1] / step) + 1 if len(t) else 0
    runs: list[RootRun] = []
    for i in range(n_bins):
        m = (t >= i * step) & (t < (i + 1) * step)
        v = pc[m]; v = v[v >= 0]
        r = int(np.bincount(v, minlength=12).argmax()) if len(v) >= max(1, 0.3 * m.sum()) else -1
        if runs and runs[-1].pc == r:
            runs[-1].dur += step
        else:
            runs.append(RootRun(round(i * step, 3), r, step))
    return [r for r in runs if r.pc >= 0 and r.dur >= min_dur]


def root_changes(runs: list[RootRun]) -> list[float]:
    out: list[float] = []
    prev = None
    for r in runs:
        if r.pc != prev:
            out.append(r.start); prev = r.pc
    return out
```

- [ ] **Step 3: Implement tempo.py**

```python
from __future__ import annotations
import librosa
import numpy as np


def _tempo_candidates(y: np.ndarray, sr: int) -> list[float]:
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
    ac = tg.mean(axis=1)
    bpms = librosa.tempo_frequencies(len(ac), sr=sr)
    idx = [i for i in np.argsort(ac)[::-1] if 30 < bpms[i] < 240][:8]
    return [float(bpms[i]) for i in idx]


def _meter(y: np.ndarray, sr: int, bpm: float) -> str:
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    hop_s = 512 / sr
    beat = 60.0 / bpm
    ac = librosa.autocorrelate(oenv)
    def at(lag_s: float) -> float:
        k = int(round(lag_s / hop_s))
        return float(ac[k]) if 0 < k < len(ac) else 0.0
    triplet = at(beat / 3) + at(2 * beat / 3)
    duple = at(beat / 2) + at(beat / 4)
    return "12/8" if triplet > duple else "4/4"


def estimate_tempo(y: np.ndarray, sr: int, root_changes: list[float]) -> tuple[float, str]:
    cands = _tempo_candidates(y, sr)
    # expand each candidate by integer factors so octave errors are recoverable
    expanded = sorted({round(c / f, 2) for c in cands for f in (1, 2, 3, 4)} | {round(c * 2, 2) for c in cands})
    expanded = [b for b in expanded if 30 <= b <= 240]
    if len(root_changes) >= 3:
        target_bar = float(np.median(np.diff(root_changes)))
        def err(b: float) -> float:
            bar = 4 * 60.0 / b
            # allow one chord per bar or per two bars
            return min(abs(bar - target_bar), abs(2 * bar - target_bar)) / target_bar
        bpm = min(expanded, key=err)
    else:
        bpm = cands[0]
    return round(bpm, 1), _meter(y, sr, bpm)
```

- [ ] **Step 4: Run, PASS, commit, push**

```bash
uv run pytest tests/test_bassroot.py tests/test_tempo.py -v
git add -A && git commit -m "Add bass root tracking and octave-safe tempo/meter estimation" && git push
```

---

### Task 8: Bar grid, chord templates, loop detection, harmony.analyze

**Files:**
- Create: `src/music_transcribe/harmony/grid.py`, `src/music_transcribe/harmony/chords.py`, modify `src/music_transcribe/harmony/__init__.py`, `tests/test_grid.py`, `tests/test_chords.py`

**Interfaces:**
- Produces:
  - `fit_grid(root_changes, bar_len) -> float` (bar0 minimizing phase error of changes to the grid).
  - `TEMPLATES: dict[str, tuple[int, ...]]` intervals, keys: `maj min 7 maj7 m7 dim m7b5 sus2 sus4 add9 9`.
  - `SUFFIX: dict[str, str]` → `{"maj":"","min":"m","7":"7","maj7":"maj7","m7":"m7","dim":"dim","m7b5":"m7b5","sus2":"sus2","sus4":"sus4","add9":"add9","9":"9"}`.
  - `chord_for_bar(chroma: np.ndarray(12), bass_pc: int | None, bar: int, start: float) -> Chord`.
  - `detect_loop(names: list[str], max_period=8, min_match=0.8) -> list[str]`.
  - `estimate_key(chroma_mean) -> str` (Krumhansl profiles; returns like `"Bbm"` or `"Db"`).
  - `analyze(stems: dict[str, Path], sr=22050) -> Harmony` in `harmony/__init__.py`.

- [ ] **Step 1: Failing tests**

`tests/test_grid.py`:
```python
from music_transcribe.harmony.grid import fit_grid


def test_fit_grid_recovers_offset():
    bar = 4.64
    changes = [0.75 + k * bar + d for k, d in zip(range(6), [0.0, 0.1, -0.1, 0.05, 0.0, -0.05])]
    assert abs(fit_grid(changes, bar) - 0.75) < 0.08
```

`tests/test_chords.py`:
```python
import numpy as np
from music_transcribe.harmony.chords import chord_for_bar, detect_loop, estimate_key, TEMPLATES


def chroma(pcs, w=1.0):
    c = np.zeros(12)
    for p in pcs: c[p % 12] = w
    return c + 0.05


def test_minor_triad_named():
    ch = chord_for_bar(chroma([10, 1, 5]), bass_pc=10, bar=1, start=0.0)
    assert ch.name == "Bbm" and ch.quality == "min" and ch.confidence > 0


def test_dominant_seventh():
    ch = chord_for_bar(chroma([5, 9, 0, 3]), bass_pc=5, bar=1, start=0.0)
    assert ch.name == "F7"


def test_inversion_uses_slash():
    ch = chord_for_bar(chroma([1, 5, 8]), bass_pc=5, bar=1, start=0.0)
    assert ch.name == "Db/F"


def test_half_diminished():
    ch = chord_for_bar(chroma([3, 6, 9, 1]), bass_pc=3, bar=1, start=0.0)
    assert ch.name == "Ebm7b5"


def test_detect_loop():
    seq = ["Bbm", "Db", "Ebm", "Bbm"] * 5 + ["Bbm", "Db"]
    assert detect_loop(seq) == ["Bbm", "Db", "Ebm", "Bbm"]
    assert detect_loop(["A", "B", "C", "D", "E", "F"]) == []


def test_estimate_key_minor():
    prof = np.zeros(12)
    for p, w in [(10, 3), (1, 2), (5, 2), (3, 1), (8, 1), (0, 0.5)]: prof[p] = w
    assert estimate_key(prof) == "Bbm"
```

- [ ] **Step 2: Implement grid.py**

```python
from __future__ import annotations
import numpy as np


def fit_grid(root_changes: list[float], bar_len: float) -> float:
    """Return bar0 in [0, bar_len) such that change times sit closest to k*bar_len + bar0."""
    if not root_changes:
        return 0.0
    ch = np.asarray(root_changes)
    # circular mean of phases
    ph = (ch % bar_len) / bar_len * 2 * np.pi
    mean = np.arctan2(np.sin(ph).mean(), np.cos(ph).mean())
    bar0 = (mean / (2 * np.pi)) * bar_len
    bar0 %= bar_len
    # shift down so bar 1 starts at or before the first change
    while bar0 > ch[0] + 1e-9:
        bar0 -= bar_len
    return float(round(bar0, 3))
```

- [ ] **Step 3: Implement chords.py**

```python
from __future__ import annotations
import numpy as np
from music_transcribe.schema import Chord, NOTE_NAMES

TEMPLATES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
    "dim": (0, 3, 6), "m7b5": (0, 3, 6, 10), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
    "add9": (0, 4, 7, 2), "9": (0, 4, 7, 10, 2),
}
SUFFIX = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "m7": "m7", "dim": "dim", "m7b5": "m7b5",
          "sus2": "sus2", "sus4": "sus4", "add9": "add9", "9": "9"}
# tetrads/pentads need a small bonus to beat their triad subset only when the extra tone is really present
_COMPLEXITY_PENALTY = {k: 0.04 * (len(v) - 3) for k, v in TEMPLATES.items()}


def _vec(root: int, quality: str) -> np.ndarray:
    v = np.zeros(12)
    for iv in TEMPLATES[quality]:
        v[(root + iv) % 12] = 1.0
    return v / np.linalg.norm(v)


def chord_for_bar(chroma: np.ndarray, bass_pc: int | None, bar: int, start: float, bass_weight: float = 0.6) -> Chord:
    c = np.asarray(chroma, dtype=float)
    c = c / (c.max() + 1e-9)
    if bass_pc is not None:
        b = np.zeros(12); b[bass_pc % 12] = 1.0
        c = c + bass_weight * b
    c = c / (np.linalg.norm(c) + 1e-9)
    scored = []
    for root in range(12):
        for q in TEMPLATES:
            s = float(np.dot(_vec(root, q), c)) - _COMPLEXITY_PENALTY[q]
            if bass_pc is not None and (bass_pc - root) % 12 not in TEMPLATES[q]:
                s -= 0.15  # bass note not a chord tone: unlikely
            scored.append((s, root, q))
    scored.sort(reverse=True)
    (s1, root, q), s2 = scored[0], scored[1][0]
    name = NOTE_NAMES[root] + SUFFIX[q]
    bass_name = NOTE_NAMES[bass_pc % 12] if bass_pc is not None else NOTE_NAMES[root]
    if bass_pc is not None and bass_pc % 12 != root:
        name += "/" + bass_name
    return Chord(bar=bar, start=float(start), name=name, root=NOTE_NAMES[root], quality=q,
                 bass=bass_name, confidence=round(max(0.0, min(1.0, (s1 - s2) * 4)), 3))


def detect_loop(names: list[str], max_period: int = 8, min_match: float = 0.8) -> list[str]:
    n = len(names)
    for p in range(2, max_period + 1):
        if n < 2 * p:
            break
        pairs = [(names[i], names[i + p]) for i in range(n - p)]
        match = sum(a == b for a, b in pairs) / len(pairs)
        if match >= min_match:
            # majority vote per position
            out = []
            for k in range(p):
                col = names[k::p]
                out.append(max(set(col), key=col.count))
            return out
    return []


_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chroma_mean: np.ndarray) -> str:
    prof = np.asarray(chroma_mean, dtype=float)
    best = None
    for i in range(12):
        for name, tmpl in (("", _MAJ), ("m", _MIN)):
            r = np.corrcoef(np.roll(tmpl, i), prof)[0, 1]
            if best is None or r > best[0]:
                best = (r, NOTE_NAMES[i] + name)
    return best[1]
```

- [ ] **Step 4: Implement harmony/__init__.py analyze()**

```python
from __future__ import annotations
from pathlib import Path
import librosa
import numpy as np
from music_transcribe.audio import load_mono
from music_transcribe.schema import Harmony, Chord
from music_transcribe.harmony.bassroot import bass_roots, root_changes
from music_transcribe.harmony.tempo import estimate_tempo
from music_transcribe.harmony.grid import fit_grid
from music_transcribe.harmony.chords import chord_for_bar, detect_loop, estimate_key

HARMONIC = ["bass", "piano", "guitar", "other"]


def analyze(stems: dict[str, Path], sr: int = 22050, mix: Path | None = None) -> Harmony:
    parts = {k: load_mono(stems[k], sr)[0] for k in HARMONIC if k in stems}
    bass = parts.get("bass")
    n = max(len(v) for v in parts.values())
    h = np.zeros(n, dtype=np.float32)
    for v in parts.values():
        h[: len(v)] += v
    full = load_mono(mix, sr)[0] if mix else h + (load_mono(stems["drums"], sr)[0] if "drums" in stems else 0)
    runs = bass_roots(bass, sr) if bass is not None else []
    changes = root_changes(runs)
    bpm, meter = estimate_tempo(full, sr, changes)
    bar_len = 4 * 60.0 / bpm
    bar0 = fit_grid(changes, bar_len)
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=h, sr=sr, hop_length=hop)
    key = estimate_key(chroma.mean(axis=1))
    # bass pc per frame from runs
    t_frames = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    bass_pc_frame = np.full(len(t_frames), -1)
    for r in runs:
        bass_pc_frame[(t_frames >= r.start) & (t_frames < r.start + r.dur)] = r.pc
    total = len(h) / sr
    chords: list[Chord] = []
    b = 0
    while bar0 + b * bar_len < total:
        t0, t1 = bar0 + b * bar_len, bar0 + (b + 1) * bar_len
        m = (t_frames >= max(0.0, t0)) & (t_frames < t1)
        if m.sum() == 0:
            b += 1; continue
        c = chroma[:, m].mean(axis=1)
        bpcs = bass_pc_frame[m]; bpcs = bpcs[bpcs >= 0]
        bass_pc = int(np.bincount(bpcs, minlength=12).argmax()) if len(bpcs) else None
        chords.append(chord_for_bar(c, bass_pc, bar=b + 1, start=round(t0, 3)))
        b += 1
    return Harmony(bpm=bpm, meter=meter, key=key, bar0=bar0, bar_len=round(bar_len, 4),
                   chords=chords, loop=detect_loop([c.name for c in chords]))
```

- [ ] **Step 5: Run, PASS, commit, push**

```bash
uv run pytest tests/test_grid.py tests/test_chords.py -v
git add -A && git commit -m "Add bar grid, chord templates with inversions, loop and key detection" && git push
```

---

### Task 9: Note detection (basic-pitch) with f0 contours and confidence

**Files:**
- Create: `src/music_transcribe/notes/__init__.py` (empty for now), `src/music_transcribe/notes/detect.py`, `src/music_transcribe/notes/confidence.py`, `tests/test_detect.py`, `tests/test_confidence.py`

**Interfaces:**
- Produces:
  - `detect.raw_notes(wav: Path, predictor=None, min_amp=0.45) -> list[Note]` — `predictor(path) -> list[tuple[start,end,pitch,amp]]` defaults to basic-pitch; applies monophonic reduction (drop a note whose onset is within 60 ms of a louder one).
  - `detect.f0_contour(y, sr, note: Note, hop=256) -> tuple[np.ndarray cents, float voiced_frac]` — pyin restricted to ±2 semitones... no: fmin/fmax = pitch ∓ 7 semitones so bends fit; cents relative to `note.pitch`, NaN where unvoiced.
  - `confidence.score(note: Note, voiced_frac: float, repeated_low_run: bool) -> tuple[str, str]`.
  - `confidence.mark_bleed_runs(notes: list[Note], low_pitch_max=52, min_run=6) -> set[int]` indices of notes in runs of ≥`min_run` consecutive identical pitches ≤ `low_pitch_max` spaced < 0.5 s.

- [ ] **Step 1: Failing tests**

`tests/test_detect.py`:
```python
import numpy as np
from music_transcribe.notes.detect import raw_notes, f0_contour
from music_transcribe.schema import Note
from conftest import sine_note, midi_to_hz, SR


def test_raw_notes_monophonic_reduction(tmp_path):
    fake = [(0.0, 0.5, 58, 0.9), (0.02, 0.5, 46, 0.3), (1.0, 1.5, 61, 0.8), (1.0, 1.2, 65, 0.2)]
    notes = raw_notes(tmp_path / "x.wav", predictor=lambda p: fake, min_amp=0.25)
    assert [n.pitch for n in notes] == [58, 61]


def test_f0_contour_flat_note_is_near_zero_cents():
    y = sine_note(midi_to_hz(58), 1.0, SR)
    cents, voiced = f0_contour(y, SR, Note(0.0, 1.0, 58, 0.9))
    assert voiced > 0.8
    assert abs(np.nanmedian(cents)) < 25


def test_f0_contour_bend_rises():
    t = np.arange(int(1.0 * SR)) / SR
    f = midi_to_hz(58) * 2 ** (np.clip((t - 0.3) / 0.4, 0, 1) * 2 / 12)   # bend up 2 semitones
    y = (0.5 * np.sin(2 * np.pi * np.cumsum(f) / SR)).astype(np.float32)
    cents, voiced = f0_contour(y, SR, Note(0.0, 1.0, 58, 0.9))
    assert np.nanmedian(cents[-len(cents)//4:]) - np.nanmedian(cents[:len(cents)//4]) > 150
```

`tests/test_confidence.py`:
```python
from music_transcribe.notes.confidence import score, mark_bleed_runs
from music_transcribe.schema import Note


def test_score_levels():
    assert score(Note(0, 1, 60, 0.9), voiced_frac=0.9, repeated_low_run=False) == ("high", "")
    assert score(Note(0, 1, 60, 0.5), voiced_frac=0.9, repeated_low_run=False) == ("medium", "weak")
    assert score(Note(0, 1, 60, 0.9), voiced_frac=0.3, repeated_low_run=False) == ("medium", "unstable")
    assert score(Note(0, 1, 46, 0.9), voiced_frac=0.9, repeated_low_run=True) == ("low", "bleed")


def test_mark_bleed_runs_finds_triplet_drone():
    notes = [Note(i * 0.3, i * 0.3 + 0.25, 46, 0.6) for i in range(8)] + [Note(3.0, 3.4, 70, 0.8)]
    idx = mark_bleed_runs(notes)
    assert idx == set(range(8))
```

- [ ] **Step 2: Implement detect.py**

```python
from __future__ import annotations
from pathlib import Path
import librosa
import numpy as np
from music_transcribe.schema import Note


def _basic_pitch_predict(path: Path) -> list[tuple[float, float, int, float]]:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, _, events = predict(str(path), ICASSP_2022_MODEL_PATH, onset_threshold=0.5, frame_threshold=0.3,
                           minimum_note_length=80, minimum_frequency=30, maximum_frequency=2000, melodia_trick=True)
    return [(float(s), float(e), int(p), float(a)) for s, e, p, a, _ in events]


def raw_notes(wav: Path, predictor=None, min_amp: float = 0.45) -> list[Note]:
    events = (predictor or _basic_pitch_predict)(wav)
    events = sorted((e for e in events if e[3] >= min_amp), key=lambda e: e[0])
    out: list[Note] = []
    for s, e, p, a in events:
        louder_overlap = any(abs(o[0] - s) < 0.06 and o[3] > a for o in events)
        if not louder_overlap:
            out.append(Note(round(s, 3), round(e, 3), p, round(a, 3)))
    return out


def f0_contour(y: np.ndarray, sr: int, note: Note, hop: int = 256) -> tuple[np.ndarray, float]:
    a, b = int(note.start * sr), int(note.end * sr)
    seg = y[a:b]
    if len(seg) < hop * 4:
        return np.array([np.nan]), 0.0
    base = librosa.midi_to_hz(note.pitch)
    fmin, fmax = base * 2 ** (-7 / 12), base * 2 ** (7 / 12)
    f0, vf, _ = librosa.pyin(seg, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop)
    cents = 1200 * np.log2(f0 / base)
    cents[~vf] = np.nan
    return cents, float(vf.mean())
```

- [ ] **Step 3: Implement confidence.py**

```python
from __future__ import annotations
from music_transcribe.schema import Note


def score(note: Note, voiced_frac: float, repeated_low_run: bool) -> tuple[str, str]:
    if repeated_low_run:
        return "low", "bleed"
    if note.amplitude < 0.35 or voiced_frac < 0.2:
        return "low", "weak" if note.amplitude < 0.35 else "unstable"
    if note.amplitude < 0.6:
        return "medium", "weak"
    if voiced_frac < 0.5:
        return "medium", "unstable"
    return "high", ""


def mark_bleed_runs(notes: list[Note], low_pitch_max: int = 52, min_run: int = 6, max_gap: float = 0.5) -> set[int]:
    flagged: set[int] = set()
    run: list[int] = []
    for i, n in enumerate(notes):
        if run and n.pitch == notes[run[-1]].pitch and n.pitch <= low_pitch_max and n.start - notes[run[-1]].end < max_gap:
            run.append(i)
        else:
            if len(run) >= min_run:
                flagged.update(run)
            run = [i] if n.pitch <= low_pitch_max else []
    if len(run) >= min_run:
        flagged.update(run)
    return flagged
```

- [ ] **Step 4: Run, PASS, commit, push**

```bash
uv run pytest tests/test_detect.py tests/test_confidence.py -v
git add -A && git commit -m "Add note detection with f0 contours and confidence scoring" && git push
```

---

### Task 10: Articulations from f0 contours

**Files:**
- Create: `src/music_transcribe/notes/articulation.py`, `tests/test_articulation.py`

**Interfaces:**
- Produces:
  - `classify_contour(cents: np.ndarray, hop_s: float) -> str` → `""`, `"bend:1"`, `"bend:2"`, `"bend-release"`, `"vibrato"`.
  - `detect_legato(prev: Note, cur: Note, gap_voiced_frac: float, onset_ratio: float) -> str` → `""`, `"slide:<cur.pitch>"` (set on prev), `"hammer"`, `"pull"` (set on cur).
  - `onset_ratio(y, sr, t, window=0.05) -> float` = onset strength at `t` divided by median onset strength.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from music_transcribe.notes.articulation import classify_contour, detect_legato
from music_transcribe.schema import Note

HOP = 256 / 22050


def ramp(n, a, b):
    return np.linspace(a, b, n)


def test_flat_is_plain():
    assert classify_contour(np.zeros(60) + np.random.randn(60) * 5, HOP) == ""


def test_full_bend():
    c = np.concatenate([np.zeros(20), ramp(20, 0, 200), np.full(30, 200)])
    assert classify_contour(c, HOP) == "bend:2"


def test_half_bend():
    c = np.concatenate([np.zeros(20), ramp(20, 0, 100), np.full(30, 100)])
    assert classify_contour(c, HOP) == "bend:1"


def test_bend_release():
    c = np.concatenate([np.zeros(10), ramp(20, 0, 200), np.full(10, 200), ramp(20, 200, 0), np.zeros(10)])
    assert classify_contour(c, HOP) == "bend-release"


def test_vibrato():
    t = np.arange(100) * HOP
    c = 40 * np.sin(2 * np.pi * 6 * t)
    assert classify_contour(c, HOP) == "vibrato"


def test_nan_heavy_is_plain():
    c = np.full(60, np.nan); c[:5] = 0
    assert classify_contour(c, HOP) == ""


def test_slide_when_gap_is_voiced():
    p, c = Note(0, 0.5, 58, 0.8), Note(0.52, 1.0, 61, 0.8)
    assert detect_legato(p, c, gap_voiced_frac=0.9, onset_ratio=0.9) == "slide:61"


def test_hammer_and_pull_on_weak_onset():
    p, c = Note(0, 0.5, 58, 0.8), Note(0.5, 1.0, 61, 0.8)
    assert detect_legato(p, c, gap_voiced_frac=0.0, onset_ratio=0.3) == "hammer"
    p2, c2 = Note(0, 0.5, 61, 0.8), Note(0.5, 1.0, 58, 0.8)
    assert detect_legato(p2, c2, gap_voiced_frac=0.0, onset_ratio=0.3) == "pull"


def test_normal_attack_is_plain():
    p, c = Note(0, 0.5, 58, 0.8), Note(0.6, 1.0, 61, 0.8)
    assert detect_legato(p, c, gap_voiced_frac=0.1, onset_ratio=1.0) == ""
```

- [ ] **Step 2: Implement**

```python
from __future__ import annotations
import librosa
import numpy as np
from music_transcribe.schema import Note

BEND_MIN_CENTS = 40.0
VIB_MIN_CENTS = 20.0
VIB_HZ = (4.0, 8.0)


def classify_contour(cents: np.ndarray, hop_s: float) -> str:
    c = np.asarray(cents, dtype=float)
    ok = ~np.isnan(c)
    if ok.sum() < max(8, 0.4 * len(c)):
        return ""
    c = c[ok]
    n = len(c)
    third = max(3, n // 3)
    head, tail = np.median(c[:third]), np.median(c[-third:])
    peak = np.max(c)
    # bend: sustained rise
    if tail - head >= BEND_MIN_CENTS:
        semis = max(1, int(round((tail - head) / 100)))
        return f"bend:{min(semis, 2)}"
    # bend-release: went up and came back
    if peak - head >= BEND_MIN_CENTS * 2 and abs(tail - head) < BEND_MIN_CENTS / 2:
        return "bend-release"
    # vibrato: periodic oscillation around a trend
    detr = c - np.convolve(c, np.ones(5) / 5, mode="same")
    if np.std(detr) >= VIB_MIN_CENTS / 2:
        crossings = np.sum(np.diff(np.sign(detr)) != 0)
        dur = n * hop_s
        hz = crossings / (2 * dur) if dur > 0 else 0
        amp = (np.percentile(c, 90) - np.percentile(c, 10)) / 2
        if VIB_HZ[0] <= hz <= VIB_HZ[1] and amp >= VIB_MIN_CENTS:
            return "vibrato"
    return ""


def detect_legato(prev: Note, cur: Note, gap_voiced_frac: float, onset_ratio: float) -> str:
    gap = cur.start - prev.end
    if gap > 0.15:
        return ""
    if gap_voiced_frac >= 0.6 and prev.pitch != cur.pitch:
        return f"slide:{cur.pitch}"
    if onset_ratio < 0.5 and abs(cur.pitch - prev.pitch) <= 4:
        return "hammer" if cur.pitch > prev.pitch else "pull"
    return ""


def onset_ratio(y: np.ndarray, sr: int, t: float, window: float = 0.05) -> float:
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=512)
    m = (times >= t - window) & (times <= t + window)
    med = np.median(oenv) + 1e-9
    return float(oenv[m].max() / med) if m.any() else 1.0
```

- [ ] **Step 3: Run, PASS, commit, push**

```bash
uv run pytest tests/test_articulation.py -v
git add -A && git commit -m "Classify bends, vibrato, slides, hammer-ons and pull-offs from f0" && git push
```

---

### Task 11: Fretboard positions and notes.transcribe_instrument

**Files:**
- Create: `src/music_transcribe/notes/fretboard.py`, modify `src/music_transcribe/notes/__init__.py`, `tests/test_fretboard.py`

**Interfaces:**
- Produces:
  - `TUNINGS = {"guitar": [64,59,55,50,45,40], "bass": [43,38,33,28]}` (string 0 = highest).
  - `assign_positions(notes: list[Note], tuning: list[int], preferred: tuple[int,int]=(3,15), max_fret=22) -> None` (sets `string`/`fret` in place; leaves None when unplayable).
  - `preferred_region(key: str) -> tuple[int, int]` → box-1 root fret of the minor pentatonic on the 6th string ±4.
  - `transcribe_instrument(wav: Path, instrument: str, harmony: Harmony, predictor=None) -> list[Note]` in `notes/__init__.py`: raw_notes → contours → articulations → legato → confidence → positions (guitar/bass only).

- [ ] **Step 1: Failing tests**

```python
from music_transcribe.notes.fretboard import assign_positions, preferred_region, TUNINGS
from music_transcribe.schema import Note


def test_positions_stay_close():
    notes = [Note(i * 0.5, i * 0.5 + 0.4, p, 0.8) for i, p in enumerate([70, 73, 75, 77, 80])]  # Bb4 Db5 Eb5 F5 Ab5
    assign_positions(notes, TUNINGS["guitar"], preferred=(6, 10))
    frets = [n.fret for n in notes]
    assert all(f is not None for f in frets)
    assert max(frets) - min(frets) <= 5


def test_unplayable_pitch_left_none():
    n = [Note(0, 1, 30, 0.8)]
    assign_positions(n, TUNINGS["guitar"])
    assert n[0].fret is None


def test_bass_tuning():
    n = [Note(0, 1, 46, 0.8)]  # Bb2
    assign_positions(n, TUNINGS["bass"])
    assert (n[0].string, n[0].fret) in {(3, 6), (2, 1)}


def test_preferred_region_for_key():
    assert preferred_region("Bbm") == (4, 10)   # Bb on 6th string = fret 6
    assert preferred_region("E") == (0, 4)
```

- [ ] **Step 2: Implement fretboard.py**

```python
from __future__ import annotations
from music_transcribe.schema import Note, NOTE_NAMES

TUNINGS: dict[str, list[int]] = {"guitar": [64, 59, 55, 50, 45, 40], "bass": [43, 38, 33, 28]}


def preferred_region(key: str) -> tuple[int, int]:
    root = key[:-1] if key.endswith("m") else key
    pc = NOTE_NAMES.index(root)
    fret = (pc - 4) % 12  # 6th string open E = pc 4
    return (max(0, fret - 2), fret + 4) if fret > 0 else (0, 4)


def _candidates(pitch: int, tuning: list[int], max_fret: int) -> list[tuple[int, int]]:
    return [(s, pitch - o) for s, o in enumerate(tuning) if 0 <= pitch - o <= max_fret]


def assign_positions(notes: list[Note], tuning: list[int], preferred: tuple[int, int] = (3, 15), max_fret: int = 22) -> None:
    prev_fret: int | None = None
    lo, hi = preferred
    for n in notes:
        cands = _candidates(n.pitch, tuning, max_fret)
        if not cands:
            n.string = n.fret = None
            continue
        def cost(c: tuple[int, int]) -> float:
            s, f = c
            jump = abs(f - prev_fret) if prev_fret is not None else abs(f - (lo + hi) / 2)
            region = 0 if lo <= f <= hi else min(abs(f - lo), abs(f - hi))
            open_pen = 0.5 if f == 0 and prev_fret and prev_fret > 3 else 0
            return jump + 1.5 * region + open_pen
        n.string, n.fret = min(cands, key=cost)
        prev_fret = n.fret
```

- [ ] **Step 3: Implement notes/__init__.py**

```python
from __future__ import annotations
from pathlib import Path
import numpy as np
from music_transcribe.audio import load_mono
from music_transcribe.schema import Note, Harmony
from music_transcribe.notes.detect import raw_notes, f0_contour
from music_transcribe.notes.articulation import classify_contour, detect_legato, onset_ratio
from music_transcribe.notes.confidence import score, mark_bleed_runs
from music_transcribe.notes.fretboard import assign_positions, preferred_region, TUNINGS

FRETTED = {"guitar", "bass"}


def _gap_voiced(y: np.ndarray, sr: int, a: float, b: float) -> float:
    import librosa
    seg = y[int(a * sr): int(b * sr)]
    if len(seg) < 2048:
        return 0.0
    _, vf, _ = librosa.pyin(seg, fmin=60, fmax=1500, sr=sr, frame_length=2048, hop_length=256)
    return float(vf.mean())


def transcribe_instrument(wav: Path, instrument: str, harmony: Harmony, predictor=None) -> list[Note]:
    y, sr = load_mono(wav, 22050)
    notes = raw_notes(wav, predictor=predictor)
    bleed = mark_bleed_runs(notes)
    hop_s = 256 / sr
    for i, n in enumerate(notes):
        cents, voiced = f0_contour(y, sr, n)
        n.articulation = classify_contour(cents, hop_s)
        n.confidence, n.reason = score(n, voiced, i in bleed)
    for i in range(1, len(notes)):
        p, c = notes[i - 1], notes[i]
        if c.start - p.end > 0.15 or p.articulation.startswith("bend"):
            continue
        art = detect_legato(p, c, _gap_voiced(y, sr, p.end, c.start) if c.start > p.end else 1.0,
                            onset_ratio(y, sr, c.start))
        if art.startswith("slide"):
            p.articulation = p.articulation or art
        elif art:
            c.articulation = c.articulation or art
    if instrument in FRETTED:
        assign_positions(notes, TUNINGS[instrument], preferred=preferred_region(harmony.key) if instrument == "guitar" else (0, 12))
    return notes
```

- [ ] **Step 4: Run, PASS, commit, push**

```bash
uv run pytest tests/test_fretboard.py -v
git add -A && git commit -m "Assign fretboard positions and wire the per-instrument note pipeline" && git push
```

---

### Task 12: Chord diagrams (shape bank + computed fallback + SVG)

**Files:**
- Create: `src/music_transcribe/render/__init__.py` (empty for now), `src/music_transcribe/render/chord_diagram.py`, `tests/test_chord_diagram.py`

**Interfaces:**
- Produces:
  - `@dataclass Voicing(frets: list[int] (6, low E first, -1 mute), base: int, barre: tuple[int,int,int] | None)`.
  - `voicing_for(name: str) -> Voicing | None` — parses `Root[quality][/bass]`; movable shapes for `maj min 7 maj7 m7 sus2 sus4 9 add9 dim m7b5` on E and A strings; bass note of a slash chord placed if reachable, else the root voicing; fallback search when no shape.
  - `svg(v: Voicing, name: str) -> str`.
  - `parse_chord(name) -> tuple[root_pc:int, quality:str, bass_pc:int|None]`.

- [ ] **Step 1: Failing tests**

```python
import pytest
from music_transcribe.render.chord_diagram import voicing_for, svg, parse_chord
from music_transcribe.harmony.chords import TEMPLATES
from music_transcribe.schema import NOTE_NAMES

OPEN = [40, 45, 50, 55, 59, 64]


def pcs(v):
    return {(o + f) % 12 for o, f in zip(OPEN, v.frets) if f >= 0}


def test_parse():
    assert parse_chord("Bbm") == (10, "min", None)
    assert parse_chord("Db/F") == (1, "maj", 5)
    assert parse_chord("Ebm7b5") == (3, "m7b5", None)
    assert parse_chord("F7") == (5, "7", None)


@pytest.mark.parametrize("name", ["Bbm", "Db", "Ebm", "F", "F7", "Ebm7b5", "Dbmaj7", "Absus4", "Bb9", "Gdim", "Db/F"])
def test_voicing_covers_chord_tones(name):
    v = voicing_for(name)
    assert v is not None, name
    root, q, _ = parse_chord(name)
    want = {(root + iv) % 12 for iv in TEMPLATES[q]}
    assert want <= pcs(v), (name, v)
    assert 0 <= v.base <= 12 and all(f == -1 or 0 <= f - (v.base - 1 if v.base > 1 else 0) <= 5 for f in v.frets)


def test_svg_has_dots_and_label():
    v = voicing_for("Bbm")
    s = svg(v, "Bbm")
    assert s.startswith("<svg") and "<circle" in s and "Bbm" in s
```

- [ ] **Step 2: Implement**

```python
from __future__ import annotations
import re
from dataclasses import dataclass
from html import escape
from music_transcribe.schema import NOTE_NAMES
from music_transcribe.harmony.chords import TEMPLATES, SUFFIX

OPEN = [40, 45, 50, 55, 59, 64]  # low E → high e
_SUFFIX_TO_Q = {v: k for k, v in SUFFIX.items()}
_QUAL_RE = sorted(_SUFFIX_TO_Q, key=len, reverse=True)

# movable shapes: offsets from the root fret; root string index (0 = low E) given by key
# E-shapes (root on string 0) and A-shapes (root on string 1). -1 = mute.
E_SHAPES = {"maj": [0, 2, 2, 1, 0, 0], "min": [0, 2, 2, 0, 0, 0], "7": [0, 2, 0, 1, 0, 0], "m7": [0, 2, 0, 0, 0, 0],
            "maj7": [0, -1, 1, 1, 0, -1], "sus4": [0, 2, 2, 2, 0, 0], "sus2": [0, 2, 4, 4, 0, 0]}
A_SHAPES = {"maj": [-1, 0, 2, 2, 2, 0], "min": [-1, 0, 2, 2, 1, 0], "7": [-1, 0, 2, 0, 2, 0], "m7": [-1, 0, 2, 0, 1, 0],
            "maj7": [-1, 0, 2, 1, 2, 0], "sus4": [-1, 0, 2, 2, 3, 0], "sus2": [-1, 0, 2, 2, 0, 0],
            "9": [-1, 0, 2, 0, 2, 2], "add9": [-1, 0, 2, 4, 2, 0], "dim": [-1, 0, 1, 2, 1, -1], "m7b5": [-1, 0, 1, 0, 1, -1]}


@dataclass
class Voicing:
    frets: list[int]
    base: int
    barre: tuple[int, int, int] | None = None  # (fret, from_string, to_string)


def parse_chord(name: str) -> tuple[int, str, int | None]:
    m = re.match(r"^([A-G][b#]?)(.*?)(?:/([A-G][b#]?))?$", name.strip())
    if not m:
        raise ValueError(name)
    root, suf, bass = m.groups()
    def pc(n: str) -> int:
        n = n.replace("#", "")
        sharp = "#" in n or False
        return NOTE_NAMES.index(n) if n in NOTE_NAMES else (NOTE_NAMES.index(n[0]) + 1) % 12
    if root not in NOTE_NAMES and "#" in root:
        root_pc = (NOTE_NAMES.index(root[0]) + 1) % 12
    else:
        root_pc = NOTE_NAMES.index(root)
    q = _SUFFIX_TO_Q.get(suf)
    if q is None:
        raise ValueError(f"unknown quality '{suf}' in {name}")
    bass_pc = None
    if bass:
        bass_pc = (NOTE_NAMES.index(bass[0]) + 1) % 12 if "#" in bass else NOTE_NAMES.index(bass)
    return root_pc, q, bass_pc


def _shape(root_pc: int, q: str) -> Voicing | None:
    options = []
    for shapes, root_string in ((E_SHAPES, 0), (A_SHAPES, 1)):
        if q not in shapes:
            continue
        root_fret = (root_pc - OPEN[root_string]) % 12
        for base in (root_fret, root_fret + 12):
            if base > 12:
                continue
            frets = [f if f < 0 else f + base for f in shapes[q]]
            if max(frets) <= 15:
                options.append((base, frets, root_string))
    if not options:
        return None
    base, frets, rs = min(options, key=lambda o: (o[0] == 0 and 0) or o[0])
    barre = None
    if base > 0:
        played = [i for i, f in enumerate(frets) if f >= 0]
        if sum(1 for i in played if frets[i] == base) >= 2:
            barre = (base, played[0], played[-1])
    return Voicing(frets, base if base > 0 else 1, barre)


def _fallback(root_pc: int, q: str) -> Voicing | None:
    tones = {(root_pc + iv) % 12 for iv in TEMPLATES[q]}
    best = None
    for base in range(1, 10):
        frets = []
        for o in OPEN:
            pick = -1
            for f in range(base, base + 4):
                if (o + f) % 12 in tones:
                    pick = f; break
            frets.append(pick)
        got = {(o + f) % 12 for o, f in zip(OPEN, frets) if f >= 0}
        if tones <= got:
            n_mute = frets.count(-1)
            if best is None or n_mute < best[0]:
                best = (n_mute, Voicing(frets, base))
    return best[1] if best else None


def voicing_for(name: str) -> Voicing | None:
    try:
        root_pc, q, bass_pc = parse_chord(name)
    except ValueError:
        return None
    v = _shape(root_pc, q) or _fallback(root_pc, q)
    if v and bass_pc is not None and bass_pc != root_pc:
        # try to put the bass note on the lowest sounding string within reach
        lo = min(range(1, 3), key=lambda s: abs(((bass_pc - OPEN[s]) % 12) - v.base))
        f = (bass_pc - OPEN[lo]) % 12
        if abs(f - v.base) <= 3:
            new = list(v.frets)
            for s in range(lo):
                new[s] = -1
            new[lo] = f
            v = Voicing(new, min(v.base, f) if f > 0 else v.base, None)
    return v


def svg(v: Voicing, name: str) -> str:
    W, H, x0, y0, sy, n = 120, 150, 22, 34, 20, 5
    sx = (W - 2 * x0) / 5
    base = v.base if v.base > 1 else 1
    o = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(name)}">']
    for i in range(6):
        x = x0 + i * sx
        o.append(f'<line class="st" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + n * sy}" stroke-width="1"/>')
    for j in range(n + 1):
        y = y0 + j * sy
        cls = "nut" if (j == 0 and base == 1) else "st"
        o.append(f'<line class="{cls}" x1="{x0}" y1="{y}" x2="{x0 + 5 * sx:.1f}" y2="{y}" stroke-width="1"/>')
    if base > 1:
        o.append(f'<text class="lbl" x="{x0 - 16}" y="{y0 + sy / 2 + 4}" text-anchor="middle">{base}</text>')
    if v.barre:
        fr, a, b = v.barre
        y = y0 + (fr - base + 0.5) * sy
        o.append(f'<rect class="brr" x="{x0 + a * sx - 7:.1f}" y="{y - 7:.1f}" width="{(b - a) * sx + 14:.1f}" height="14" rx="7"/>')
    for i, f in enumerate(v.frets):
        x = x0 + i * sx
        if f == -1:
            o.append(f'<text class="lbl" x="{x:.1f}" y="{y0 - 8}" text-anchor="middle">×</text>')
        elif f == 0:
            o.append(f'<circle cx="{x:.1f}" cy="{y0 - 11}" r="4" fill="none" class="st" stroke-width="1"/>')
        else:
            y = y0 + (f - base + 0.5) * sy
            o.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="7"/>')
    o.append("</svg>")
    return "".join(o)
```

Note for the implementer: `parse_chord` contains a dead inner `pc` helper; delete it. The test's fret-span assertion uses `v.base - 1 if v.base > 1 else 0`; keep `Voicing.base` as the fret number shown at the top-left (1 for open position).

- [ ] **Step 3: Run, PASS, commit, push**

```bash
uv run pytest tests/test_chord_diagram.py -v
git add -A && git commit -m "Generate chord diagrams from movable shapes with computed fallback" && git push
```

---

### Task 13: ASCII tab, cifra.txt and MIDI

**Files:**
- Create: `src/music_transcribe/render/tab.py`, `src/music_transcribe/render/text.py`, `src/music_transcribe/render/midi.py`, `tests/test_tab.py`, `tests/test_text.py`, `tests/test_midi.py`

**Interfaces:**
- Produces:
  - `tab.STRING_NAMES = {"guitar": "eBGDAE", "bass": "GDAE"}`; `tab.token(note) -> str` (`"7"`, `"7b9"`, `"7~"`, `"7/9"`, `"7h"`, `"7p"`, bend target = fret + semitones); `tab.bar_tab(notes, bar_start, bar_len, meter, instrument) -> str` (12 cells for 12/8, 16 for 4/4, cell width 3, beat separators `|`); `tab.bars_for(notes, harmony) -> list[tuple[int, str]]` non-empty bars only.
  - `text.cifra_txt(tags, harmony, lyrics, notes_by_inst) -> str`.
  - `midi.write_midi(notes, path, program=27)`.

- [ ] **Step 1: Failing tests**

`tests/test_tab.py`:
```python
from music_transcribe.render.tab import token, bar_tab
from music_transcribe.schema import Note


def n(start, pitch, fret, string, art="", dur=0.3):
    return Note(start, start + dur, pitch, 0.8, articulation=art, string=string, fret=fret)


def test_tokens():
    assert token(n(0, 70, 6, 0)) == "6"
    assert token(n(0, 70, 6, 0, "bend:2")) == "6b8"
    assert token(n(0, 70, 6, 0, "bend:1")) == "6b7"
    assert token(n(0, 70, 6, 0, "bend-release")) == "6br"
    assert token(n(0, 70, 6, 0, "vibrato")) == "6~"
    assert token(n(0, 70, 6, 0, "slide:73")) == "6/9"
    assert token(n(0, 70, 6, 0, "hammer")) == "h6"
    assert token(n(0, 70, 6, 0, "pull")) == "p6"


def test_bar_tab_12_8_layout():
    bar = 4.8; notes = [n(0.0, 70, 6, 0), n(1.2, 65, 6, 1, "vibrato"), n(2.4, 58, 8, 3)]
    s = bar_tab(notes, 0.0, bar, "12/8", "guitar")
    lines = s.splitlines()
    assert len(lines) == 6 and lines[0].startswith("e|") and lines[5].startswith("E|")
    assert lines[0].count("|") == 6            # name sep + 4 beats + closing
    assert "6  " in lines[0] and "6~ " in lines[1] and "8  " in lines[3]


def test_bar_tab_4_4_has_16_cells():
    s = bar_tab([n(0.0, 70, 6, 0)], 0.0, 2.0, "4/4", "guitar")
    assert len(s.splitlines()[0]) == 2 + 16 * 3 + 5
```

`tests/test_text.py`:
```python
from music_transcribe.render.text import cifra_txt
from music_transcribe.schema import Tags, Harmony, Chord, LyricLine, Note


def test_cifra_txt_sections():
    h = Harmony(52, "12/8", "Bbm", 0.75, 4.64, [Chord(1, 0.75, "Bbm", "Bb", "min", "Bb", 0.4)], ["Bbm", "Db", "Ebm", "Bbm"])
    txt = cifra_txt(Tags(title="T", artist="A"), h, [LyricLine(20, 25, "Moonlight", 0.9)], {"guitar": [Note(1.0, 1.3, 70, 0.8, string=0, fret=6)]})
    assert "T — A" in txt and "Bbm" in txt and "Moonlight" in txt and "e|" in txt and "12/8" in txt
```

`tests/test_midi.py`:
```python
import pretty_midi
from music_transcribe.render.midi import write_midi
from music_transcribe.schema import Note


def test_write_midi(tmp_path):
    p = tmp_path / "g.mid"
    write_midi([Note(0.0, 0.5, 58, 0.9), Note(0.5, 1.0, 61, 0.4)], p)
    pm = pretty_midi.PrettyMIDI(str(p))
    assert [n.pitch for n in pm.instruments[0].notes] == [58, 61]
```

- [ ] **Step 2: Implement tab.py**

```python
from __future__ import annotations
from music_transcribe.schema import Note, Harmony

STRING_NAMES = {"guitar": "eBGDAE", "bass": "GDAE"}
CELL = 3


def token(n: Note) -> str:
    f = "" if n.fret is None else str(n.fret)
    a = n.articulation
    if a.startswith("bend:") and n.fret is not None:
        return f"{f}b{n.fret + int(a.split(':')[1])}"
    if a == "bend-release":
        return f"{f}br"
    if a == "vibrato":
        return f"{f}~"
    if a.startswith("slide:") and n.fret is not None and n.string is not None:
        # target fret on the same string
        return f"{f}/{n.fret + (int(a.split(':')[1]) - n.pitch)}"
    if a == "hammer":
        return f"h{f}"
    if a == "pull":
        return f"p{f}"
    return f


def bar_tab(notes: list[Note], bar_start: float, bar_len: float, meter: str, instrument: str) -> str:
    names = STRING_NAMES[instrument]
    cells = 12 if meter == "12/8" else 16
    per_beat = cells // 4
    sub = bar_len / cells
    grid: list[list[str]] = [["-" * CELL] * cells for _ in names]
    for n in sorted(notes, key=lambda x: x.start):
        if n.string is None or n.fret is None or not (bar_start <= n.start < bar_start + bar_len):
            continue
        k = min(cells - 1, int(round((n.start - bar_start) / sub)))
        t = token(n)[:CELL].ljust(CELL, "-")
        grid[n.string][k] = t
    lines = []
    for s, name in enumerate(names):
        row = "".join(("|" if k % per_beat == 0 else "") + grid[s][k] for k in range(cells))
        lines.append(f"{name}|{row}|")
    return "\n".join(lines)


def bars_for(notes: list[Note], harmony: Harmony, instrument: str) -> list[tuple[int, str]]:
    if not notes:
        return []
    last = max(n.start for n in notes)
    out = []
    b = 0
    while harmony.bar0 + b * harmony.bar_len <= last:
        t0 = harmony.bar0 + b * harmony.bar_len
        in_bar = [n for n in notes if t0 <= n.start < t0 + harmony.bar_len and n.fret is not None]
        if in_bar:
            out.append((b + 1, bar_tab(in_bar, t0, harmony.bar_len, harmony.meter, instrument)))
        b += 1
    return out
```

Adjust the 4/4 test expectation: line = name(1) + "|"(1) + cells*3 + 4 beat separators + closing "|" = 2 + 48 + 5 = 55, which the test already encodes.

- [ ] **Step 3: Implement text.py and midi.py**

`text.py`:
```python
from __future__ import annotations
from music_transcribe.schema import Tags, Harmony, LyricLine, Note
from music_transcribe.render.tab import bars_for, STRING_NAMES


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def cifra_txt(tags: Tags, harmony: Harmony, lyrics: list[LyricLine], notes_by_inst: dict[str, list[Note]]) -> str:
    out = [f"{tags.title or 'Sem título'} — {tags.artist or 'Artista desconhecido'}",
           f"Tom: {harmony.key} | {harmony.bpm:.0f} BPM em {harmony.meter} | compasso = {harmony.bar_len:.2f} s", ""]
    if harmony.loop:
        out += ["PROGRESSÃO (loop): | " + " | ".join(harmony.loop) + " |", ""]
    out.append("ACORDES POR COMPASSO")
    for c in harmony.chords:
        flag = "" if c.confidence >= 0.3 else "  (?)"
        out.append(f"{c.bar:4d} {_mmss(c.start)}  {c.name}{flag}")
    out += ["", "LETRA"]
    if not lyrics:
        out.append("(instrumental)")
    for l in lyrics:
        out.append(f"[{_mmss(l.start)}] {l.text}{'' if l.confidence >= 0.6 else ' (?)'}")
    for inst, notes in notes_by_inst.items():
        if inst not in STRING_NAMES:
            continue
        out += ["", f"TAB — {inst.upper()}  (b=bend, br=bend-release, ~=vibrato, /=slide, h=hammer, p=pull)"]
        for bar, tab in bars_for(notes, harmony, inst):
            chord = next((c.name for c in harmony.chords if c.bar == bar), "")
            low = sum(1 for n in notes if n.confidence == "low" and harmony.bar0 + (bar - 1) * harmony.bar_len <= n.start < harmony.bar0 + bar * harmony.bar_len)
            out += [f"compasso {bar}  {chord}" + (f"  [{low} nota(s) confiança baixa]" if low else ""), tab, ""]
    return "\n".join(out) + "\n"
```

`midi.py`:
```python
from __future__ import annotations
from pathlib import Path
import pretty_midi
from music_transcribe.schema import Note


def write_midi(notes: list[Note], path: Path, program: int = 27) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for n in notes:
        inst.notes.append(pretty_midi.Note(velocity=int(40 + 80 * min(1.0, n.amplitude)), pitch=int(n.pitch),
                                           start=float(n.start), end=float(max(n.end, n.start + 0.05))))
    pm.instruments.append(inst)
    pm.write(str(path))
```

- [ ] **Step 4: Run, PASS, commit, push**

```bash
uv run pytest tests/test_tab.py tests/test_text.py tests/test_midi.py -v
git add -A && git commit -m "Render ASCII tab with articulations, cifra.txt and MIDI" && git push
```

---

### Task 14: HTML page

**Files:**
- Create: `src/music_transcribe/render/html.py`, modify `src/music_transcribe/render/__init__.py`, `tests/test_html.py`

**Interfaces:**
- Produces: `html.cifra_html(tags, harmony, lyrics, notes_by_inst) -> str`; `render.render_all(out: OutDir) -> dict[str, Path]` reading `tags/tags.json`, `harmony/harmony.json`, `lyrics/lyrics.json`, `notes/<inst>.json` and writing `render/cifra.html`, `render/cifra.txt`, `render/<inst>.mid`.

- [ ] **Step 1: Failing tests**

```python
from music_transcribe.render.html import cifra_html
from music_transcribe.schema import Tags, Harmony, Chord, LyricLine, Note

H = Harmony(52, "12/8", "Bbm", 0.75, 4.64,
            [Chord(1, 0.75, "Bbm", "Bb", "min", "Bb", 0.4), Chord(2, 5.39, "Db", "Db", "maj", "Db", 0.1)],
            ["Bbm", "Db", "Ebm", "Bbm"])


def test_html_sections_and_confidence_markers():
    html = cifra_html(Tags(title="Midnight", artist="CBR"), H,
                      [LyricLine(20, 25, "Moonlight", 0.9), LyricLine(25, 30, "For another glance", 0.4)],
                      {"guitar": [Note(1.0, 1.3, 70, 0.8, articulation="bend:2", confidence="low", reason="bleed", string=0, fret=6)]})
    assert html.startswith("<title>Midnight</title>")
    assert 'class="chords-grid"' in html and "<svg" in html
    assert "Moonlight" in html and 'class="low"' in html            # low-confidence lyric line marked
    assert "6b8" in html and "conf-low" in html                      # tab token and note confidence class
    assert "prefers-color-scheme: dark" in html and 'data-theme="dark"' in html


def test_html_instrumental():
    html = cifra_html(Tags(), H, [], {})
    assert "instrumental" in html.lower()
```

- [ ] **Step 2: Implement html.py**

```python
from __future__ import annotations
from html import escape
from music_transcribe.schema import Tags, Harmony, LyricLine, Note
from music_transcribe.render.chord_diagram import voicing_for, svg
from music_transcribe.render.tab import bars_for, STRING_NAMES

CSS = """
:root{--bg:#f6f3ee;--ink:#1c1a1f;--muted:#6b6470;--rule:#dcd6cc;--panel:#fbf9f5;--accent:#b8651a;--accent-ink:#7a4110;--chord:#2a5f8a;--pre:#fffdf9;--low:#b3261e;--mid:#8a6d00}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15131a;--ink:#ece7df;--muted:#9d95a5;--rule:#2f2b37;--panel:#1d1a24;--accent:#e08a3a;--accent-ink:#f2b06d;--chord:#8fc1ec;--pre:#100f14;--low:#ff8a80;--mid:#e6c34a}}
:root[data-theme="dark"]{--bg:#15131a;--ink:#ece7df;--muted:#9d95a5;--rule:#2f2b37;--panel:#1d1a24;--accent:#e08a3a;--accent-ink:#f2b06d;--chord:#8fc1ec;--pre:#100f14;--low:#ff8a80;--mid:#e6c34a}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:16px;line-height:1.5;padding-inline:clamp(16px,4vw,48px);padding-block:32px 64px}
.wrap{max-width:1040px;margin:0 auto;display:grid;gap:40px}
h1,h2{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;text-wrap:balance;margin:0}
h1{font-size:clamp(34px,6vw,58px);line-height:1;font-weight:700;letter-spacing:-.02em}
h2{font-size:22px;font-weight:700;padding-bottom:8px;border-bottom:2px solid var(--rule);margin-bottom:16px}
.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
header{display:grid;gap:10px}
.meta{display:flex;flex-wrap:wrap;gap:8px 20px;font-family:"IBM Plex Mono",monospace;font-size:14px;color:var(--muted)}
.meta b{color:var(--ink);font-weight:500}
.prog{display:flex;flex-wrap:wrap;gap:8px;font-family:"IBM Plex Mono",monospace;font-size:18px}
.prog span{background:var(--panel);border:1px solid var(--rule);padding:8px 14px;border-radius:4px;font-weight:500;color:var(--chord)}
.chords-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;max-width:760px}
.cd{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:12px 10px 8px;display:grid;justify-items:center;gap:4px}
.cd .nm{font-family:"IBM Plex Mono",monospace;font-size:20px;font-weight:500;color:var(--chord)}
.cd svg{width:120px;height:auto;max-width:100%}
.cd .st{stroke:var(--ink)}.cd .dot{fill:var(--ink)}.cd .lbl{fill:var(--muted);font:11px "IBM Plex Mono",monospace}.cd .nut{stroke:var(--ink);stroke-width:4}.cd .brr{fill:var(--ink);opacity:.9}
.cols{display:grid;grid-template-columns:1fr;gap:32px}
@media (min-width:760px){.cols{grid-template-columns:1fr 1fr}}
.line{display:grid;grid-template-columns:auto 1fr;gap:12px;align-items:baseline;padding:4px 0}
.line time{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted)}
.line.low{color:var(--low)} .line.low::after{content:" ?";color:var(--low)}
.note{background:var(--panel);border-left:3px solid var(--accent);padding:12px 16px;font-size:14px;color:var(--muted);max-width:60em}
.bars{display:grid;grid-template-columns:1fr;gap:18px}
@media (min-width:760px){.bars{grid-template-columns:1fr 1fr}}
.bar{margin:0;background:var(--panel);border:1px solid var(--rule);border-radius:4px;overflow:hidden}
.bar figcaption{display:flex;gap:12px;align-items:baseline;padding:8px 12px;border-bottom:1px solid var(--rule);font-family:"IBM Plex Mono",monospace;font-size:13px}
.bar .chord{color:var(--chord);font-weight:500}.bar .time{margin-left:auto;color:var(--muted)}
.bar pre{margin:0;padding:10px 12px;background:var(--pre);font-family:"IBM Plex Mono",monospace;font-size:13.5px;line-height:1.35;overflow-x:auto;color:var(--ink)}
.conf-low{color:var(--low)}.conf-medium{color:var(--mid)}
.legend{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--muted);margin-bottom:14px}
.chordtag{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--chord)}
"""
FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">'


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _chord_cards(names: list[str]) -> str:
    cards = []
    for n in names:
        v = voicing_for(n)
        body = svg(v, n) if v else '<div class="legend">sem diagrama</div>'
        cards.append(f'<div class="cd"><div class="nm">{escape(n)}</div>{body}</div>')
    return '<div class="chords-grid">' + "".join(cards) + "</div>"


def _chord_at(harmony: Harmony, t: float) -> str:
    for c in reversed(harmony.chords):
        if c.start <= t:
            return c.name
    return ""


def _lyrics(harmony: Harmony, lyrics: list[LyricLine]) -> str:
    if not lyrics:
        return '<p class="legend">Instrumental — nenhuma voz detectada.</p>'
    rows = []
    for l in lyrics:
        cls = "line low" if l.confidence < 0.6 else "line"
        rows.append(f'<div class="{cls}"><time>{_mmss(l.start)}</time><div><span class="chordtag">{escape(_chord_at(harmony, l.start))}</span><br>{escape(l.text)}</div></div>')
    return "".join(rows)


def _tab_section(inst: str, notes: list[Note], harmony: Harmony) -> str:
    bars = bars_for(notes, harmony, inst)
    if not bars:
        return ""
    figs = []
    for bar, tab in bars:
        t0 = harmony.bar0 + (bar - 1) * harmony.bar_len
        chord = next((c.name for c in harmony.chords if c.bar == bar), "")
        in_bar = [n for n in notes if t0 <= n.start < t0 + harmony.bar_len]
        worst = "low" if any(n.confidence == "low" for n in in_bar) else ("medium" if any(n.confidence == "medium" for n in in_bar) else "high")
        reasons = sorted({n.reason for n in in_bar if n.reason})
        figs.append(f'<figure class="bar"><figcaption><span>Compasso {bar}</span><span class="chord">{escape(chord)}</span>'
                    f'<span class="conf-{worst}">{worst}{" · " + ", ".join(reasons) if reasons else ""}</span><span class="time">{_mmss(t0)}</span></figcaption>'
                    f'<pre>{escape(tab)}</pre></figure>')
    return (f'<section><h2>{escape(inst.capitalize())} · tab</h2>'
            f'<div class="legend">{harmony.meter} · cada "|" é um tempo · b=bend (casa alvo) · br=bend-release · ~=vibrato · /=slide · h=hammer · p=pull · cor = confiança</div>'
            f'<div class="bars">{"".join(figs)}</div></section>')


def cifra_html(tags: Tags, harmony: Harmony, lyrics: list[LyricLine], notes_by_inst: dict[str, list[Note]]) -> str:
    title = tags.title or "Sem título"
    uniq = list(dict.fromkeys(harmony.loop or [c.name for c in harmony.chords]))
    low_chords = sum(1 for c in harmony.chords if c.confidence < 0.3)
    parts = [f"<title>{escape(title)}</title>", FONTS, f"<style>{CSS}</style>", '<div class="wrap">',
             "<header>", f'<div class="eyebrow">{escape(tags.artist)}{" · " + escape(tags.album) if tags.album else ""}</div>',
             f"<h1>{escape(title)}</h1>",
             f'<div class="meta"><span>Tom <b>{escape(harmony.key)}</b></span><span>Andamento <b>{harmony.bpm:.0f} BPM · {harmony.meter}</b></span><span>Compasso <b>{harmony.bar_len:.2f} s</b></span></div>']
    if harmony.loop:
        parts.append('<div class="prog">' + "".join(f"<span>{escape(n)}</span>" for n in harmony.loop) + "</div>")
    parts.append(_chord_cards(uniq))
    parts.append(f'<div class="note">Acordes por compasso: {len(harmony.chords)}; {low_chords} com confiança baixa (marcados na lista).</div></header>')
    parts.append('<div class="cols"><section><h2>Letra</h2>' + _lyrics(harmony, lyrics) + "</section>")
    parts.append('<section><h2>Acordes por compasso</h2><div class="legend">' + " ".join(
        f'<span class="{"conf-low" if c.confidence < 0.3 else ""}">{c.bar}:{escape(c.name)}</span>' for c in harmony.chords) + "</div></section></div>")
    for inst in ("guitar", "bass"):
        if inst in notes_by_inst:
            parts.append(_tab_section(inst, notes_by_inst[inst], harmony))
    if "piano" in notes_by_inst and notes_by_inst["piano"]:
        rows = "".join(f"<div class='line conf-{n.confidence}'><time>{_mmss(n.start)}</time><div>{escape(__import__('music_transcribe.schema', fromlist=['pitch_name']).pitch_name(n.pitch))}</div></div>" for n in notes_by_inst["piano"])
        parts.append(f"<section><h2>Piano · notas</h2>{rows}</section>")
    parts.append("</div>")
    return "\n".join(parts)
```

Replace the inline `__import__` with a top-level `from music_transcribe.schema import pitch_name`.

- [ ] **Step 3: Implement render/__init__.py**

```python
from __future__ import annotations
from pathlib import Path
from music_transcribe.cache import OutDir
from music_transcribe.schema import Tags, Harmony, LyricLine, Note, load_json
from music_transcribe.render.html import cifra_html
from music_transcribe.render.text import cifra_txt
from music_transcribe.render.midi import write_midi

PROGRAMS = {"guitar": 27, "bass": 33, "piano": 0}


def render_all(out: OutDir) -> dict[str, Path]:
    tags = load_json(Tags, out.stage("tags") / "tags.json") if (out.stage("tags") / "tags.json").exists() else Tags()
    harmony = load_json(Harmony, out.stage("harmony") / "harmony.json")
    lp = out.stage("lyrics") / "lyrics.json"
    lyrics = load_json(LyricLine, lp) if lp.exists() else []
    notes_by_inst = {p.stem: load_json(Note, p) for p in sorted(out.stage("notes").glob("*.json"))}
    r = out.stage("render")
    files = {"html": r / "cifra.html", "txt": r / "cifra.txt"}
    files["html"].write_text(cifra_html(tags, harmony, lyrics, notes_by_inst))
    files["txt"].write_text(cifra_txt(tags, harmony, lyrics, notes_by_inst))
    for inst, notes in notes_by_inst.items():
        p = r / f"{inst}.mid"
        write_midi(notes, p, PROGRAMS.get(inst, 0))
        files[inst] = p
    out.mark_done("render")
    return files
```

- [ ] **Step 4: Run, PASS, commit, push**

```bash
uv run pytest tests/test_html.py -v
git add -A && git commit -m "Render the cifra HTML page and wire render_all" && git push
```

---

### Task 15: CLI `run` and per-stage subcommands

**Files:**
- Modify: `src/music_transcribe/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes everything above.
- Produces commands: `run AUDIO [--out DIR] [--force] [--yes] [--model large-v3] [--instruments guitar,bass,piano] [--device mps]`, and `tags`, `stems`, `lyrics`, `harmony`, `notes`, `render`, each `AUDIO [--out DIR] [--force]`. Default `--out` = `<audio dir>/<audio stem>.transcribe/`. `--yes` auto-confirms model downloads; otherwise `typer.confirm`.
- Internal: `pipeline.py` functions `stage_tags(audio, out)`, `stage_stems(audio, out, device)`, `stage_lyrics(audio, out, model, confirm)`, `stage_harmony(audio, out)`, `stage_notes(audio, out, instruments)`, `stage_render(out)`; each skips when `out.done(name)` and not `force`, and writes its JSON.

- [ ] **Step 1: Failing test (stage orchestration with all heavy work stubbed)**

Append to `tests/test_cli.py`:
```python
from pathlib import Path
import json
from music_transcribe import pipeline
from music_transcribe.schema import Tags, Harmony, Chord, LyricLine, Note, save_json


def test_run_skips_done_stages_and_renders(tmp_path, monkeypatch):
    audio = tmp_path / "song.flac"; audio.write_bytes(b"x")
    out = tmp_path / "out"
    calls = []
    monkeypatch.setattr(pipeline, "read_tags", lambda p, **k: calls.append("tags") or Tags(title="T", artist="A"))
    monkeypatch.setattr(pipeline, "separate", lambda a, o, **k: calls.append("stems") or {s: (Path(o) / f"{s}.wav") for s in ["vocals", "guitar", "bass", "drums", "piano", "other"]})
    monkeypatch.setattr(pipeline, "to_wav", lambda s, d, **k: Path(d).write_bytes(b"RIFF") or Path(d))
    monkeypatch.setattr(pipeline, "ensure_whisper_model", lambda n, confirm, **k: Path("/m.bin"))
    monkeypatch.setattr(pipeline, "transcribe", lambda w, m, **k: calls.append("lyrics") or [LyricLine(1, 2, "hi", 0.9)])
    monkeypatch.setattr(pipeline, "analyze", lambda stems, **k: calls.append("harmony") or Harmony(100, "4/4", "C", 0.0, 2.4, [Chord(1, 0.0, "C", "C", "maj", "C", 0.5)], []))
    monkeypatch.setattr(pipeline, "transcribe_instrument", lambda w, inst, h, **k: calls.append(f"notes:{inst}") or [Note(0.1, 0.4, 60, 0.9, string=1, fret=1)])
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    assert r.exit_code == 0, r.output
    assert (out / "render" / "cifra.html").exists() and (out / "render" / "guitar.mid").exists()
    assert calls == ["tags", "stems", "lyrics", "harmony", "notes:guitar"]
    calls.clear()
    r = runner.invoke(app, ["run", str(audio), "--out", str(out), "--yes", "--instruments", "guitar"])
    assert r.exit_code == 0 and calls == []          # everything cached
    r = runner.invoke(app, ["harmony", str(audio), "--out", str(out), "--force"])
    assert r.exit_code == 0 and calls == ["harmony"]
```

- [ ] **Step 2: Implement pipeline.py**

`src/music_transcribe/pipeline.py`:
```python
from __future__ import annotations
from pathlib import Path
from typing import Callable
from music_transcribe.cache import OutDir
from music_transcribe.schema import save_json, load_json, Harmony
from music_transcribe.tags import read_tags
from music_transcribe.stems import separate, STEM_NAMES
from music_transcribe.audio import to_wav
from music_transcribe.models import ensure_whisper_model
from music_transcribe.lyrics import transcribe
from music_transcribe.harmony import analyze
from music_transcribe.notes import transcribe_instrument
from music_transcribe.render import render_all


def _skip(out: OutDir, name: str, force: bool) -> bool:
    return out.done(name) and not force


def stage_tags(audio: Path, out: OutDir, force: bool = False) -> None:
    if _skip(out, "tags", force):
        return
    save_json(read_tags(audio), out.stage("tags") / "tags.json")
    out.mark_done("tags")


def stage_stems(audio: Path, out: OutDir, force: bool = False, device: str = "mps") -> dict[str, Path]:
    d = out.stage("stems")
    if not _skip(out, "stems", force):
        wav = to_wav(audio, d / "_input.wav")
        separate(wav, d, device=device)
        wav.unlink(missing_ok=True)
        out.mark_done("stems")
    return {s: d / f"{s}.wav" for s in STEM_NAMES}


def stage_lyrics(audio: Path, out: OutDir, model: str, confirm: Callable[[str, int], bool], force: bool = False) -> None:
    if _skip(out, "lyrics", force):
        return
    stems = stage_stems(audio, out)
    d = out.stage("lyrics")
    v16 = to_wav(stems["vocals"], d / "vocals16.wav", sr=16000, mono=True)
    mp = ensure_whisper_model(model, confirm) or ensure_whisper_model("medium.en", confirm)
    if mp is None:
        raise RuntimeError("nenhum modelo Whisper disponível; rode com --yes ou baixe manualmente")
    save_json(transcribe(v16, mp), d / "lyrics.json")
    out.mark_done("lyrics")


def stage_harmony(audio: Path, out: OutDir, force: bool = False) -> Harmony:
    d = out.stage("harmony")
    if _skip(out, "harmony", force):
        return load_json(Harmony, d / "harmony.json")
    stems = stage_stems(audio, out)
    h = analyze(stems)
    save_json(h, d / "harmony.json")
    out.mark_done("harmony")
    return h


def stage_notes(audio: Path, out: OutDir, instruments: list[str], force: bool = False) -> None:
    if _skip(out, "notes", force):
        return
    stems = stage_stems(audio, out)
    h = stage_harmony(audio, out)
    d = out.stage("notes")
    for inst in instruments:
        save_json(transcribe_instrument(stems[inst], inst, h), d / f"{inst}.json")
    out.mark_done("notes")


def stage_render(out: OutDir) -> dict[str, Path]:
    return render_all(out)
```

- [ ] **Step 3: Implement cli.py commands**

Replace `cli.py` with:
```python
from __future__ import annotations
from pathlib import Path
import typer
from music_transcribe import __version__, pipeline
from music_transcribe.cache import OutDir

app = typer.Typer(help="Letra, cifra e tab de uma música a partir do áudio, localmente.", no_args_is_help=True)


def _version(value: bool):
    if value:
        typer.echo(__version__); raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version, is_eager=True)):
    """music-transcribe"""


def _out(audio: Path, out: Path | None) -> OutDir:
    return OutDir(out or audio.parent / f"{audio.stem}.transcribe")


def _confirm(yes: bool):
    def f(name: str, mb: int) -> bool:
        return yes or typer.confirm(f"Baixar modelo Whisper {name} (~{mb} MB) para o cache local?")
    return f


AUDIO = typer.Argument(..., exists=True, dir_okay=False)
OUT = typer.Option(None, "--out", help="Pasta de saída (padrão: <audio>.transcribe/)")
FORCE = typer.Option(False, "--force", help="Refaz a etapa mesmo com cache")


@app.command()
def run(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, yes: bool = typer.Option(False, "--yes"),
        model: str = typer.Option("large-v3", "--model"), instruments: str = typer.Option("guitar,bass,piano", "--instruments"),
        device: str = typer.Option("mps", "--device")):
    """Roda o pipeline inteiro: tags → stems → lyrics → harmony → notes → render."""
    o = _out(audio, out)
    pipeline.stage_tags(audio, o, force)
    pipeline.stage_stems(audio, o, force, device)
    pipeline.stage_lyrics(audio, o, model, _confirm(yes), force)
    pipeline.stage_harmony(audio, o, force)
    pipeline.stage_notes(audio, o, [i.strip() for i in instruments.split(",") if i.strip()], force)
    files = pipeline.stage_render(o)
    for k, p in files.items():
        typer.echo(f"{k}: {p}")


@app.command()
def tags(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE):
    pipeline.stage_tags(audio, _out(audio, out), force)


@app.command()
def stems(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, device: str = typer.Option("mps", "--device")):
    for k, p in pipeline.stage_stems(audio, _out(audio, out), force, device).items():
        typer.echo(f"{k}: {p}")


@app.command()
def lyrics(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE, yes: bool = typer.Option(False, "--yes"),
           model: str = typer.Option("large-v3", "--model")):
    pipeline.stage_lyrics(audio, _out(audio, out), model, _confirm(yes), force)


@app.command()
def harmony(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE):
    h = pipeline.stage_harmony(audio, _out(audio, out), force)
    typer.echo(f"{h.key} · {h.bpm} BPM · {h.meter} · loop {h.loop}")


@app.command()
def notes(audio: Path = AUDIO, out: Path | None = OUT, force: bool = FORCE,
          instruments: str = typer.Option("guitar,bass,piano", "--instruments")):
    pipeline.stage_notes(audio, _out(audio, out), [i.strip() for i in instruments.split(",")], force)


@app.command()
def render(audio: Path = AUDIO, out: Path | None = OUT):
    for k, p in pipeline.stage_render(_out(audio, out)).items():
        typer.echo(f"{k}: {p}")
```

- [ ] **Step 4: Run all fast tests, PASS, commit, push**

```bash
uv run pytest -v
git add -A && git commit -m "Wire the staged pipeline into the CLI with caching and --force" && git push
```

---

### Task 16: Skill `transcribing-a-song`, README

**Files:**
- Create: `skills/transcribing-a-song/SKILL.md`, `README.md`

**REQUIRED SUB-SKILL:** authoring a `SKILL.md` requires invoking `superpowers:writing-skills` first (baseline with a subagent, then write, then verify). The content below is the target; the writing-skills process decides final wording.

- [ ] **Step 1: Invoke `superpowers:writing-skills`** and run its baseline: give a subagent the task "the user has `~/Music/x.flac` and asks for the cifra" with no skill loaded and record what it gets wrong (expected: tries online lyrics, skips confidence, doesn't run the CLI).

- [ ] **Step 2: Write SKILL.md**

```markdown
---
name: transcribing-a-song
description: Use when the user asks for the lyrics, chords, cifra, tab or solo of a song they have as an audio file (flac, wav, mp3, m4a) — runs the local music-transcribe pipeline, reviews confidence, publishes the page and copies the folder where the user keeps transcriptions.
---

# Transcribing a song from audio

## Overview
The user has the audio and wants to play along. Everything runs locally: `music-transcribe run <audio>` separates stems, transcribes vocals with Whisper, estimates key/tempo/chords per bar, transcribes notes with articulations, and renders `cifra.html`, `cifra.txt` and MIDI. Your job is to run it, **review** what it produced, and deliver it where the user keeps music.

## When to use
- "letra dessa música", "cifra", "acordes", "tab do solo", "extrai do flac" with a local file.
- Not for songs the user does not have as a file (no downloading audio).

## Steps
1. Locate the file; confirm it exists. Default output: `<audio>.transcribe/`.
2. Run `music-transcribe run "<audio>" --yes` (the first run downloads Whisper large-v3, ~3 GB; say so in one line before).
3. Open `harmony/harmony.json`: sanity-check `bpm` (slow blues 40–70, pop 80–130), `meter`, `key`, `loop`. If `loop` is empty on a song that clearly repeats, re-run `music-transcribe harmony --force` after checking the bass stem is not silent.
4. Open `lyrics/lyrics.json`: lines with `confidence < 0.6` get a `?` in your summary. Never "fix" a lyric from memory; say it is uncertain.
5. Open `notes/*.json`: report how many notes are `low` and why (`bleed`, `weak`, `unstable`). Bars where every note is `low` are probably accompaniment, not the solo — say so.
6. Publish `render/cifra.html` as an artifact (load `artifact-design` first). Copy the output folder to the user's transcription folder when they name one (João: iCloud `Musica/Transcricoes/<Artista - Título>/`).
7. Deliver in ≤4 lines: key/tempo/loop, count of uncertain lyric lines, count of low-confidence tab bars, links.

## Rules
- Confidence is shown, never hidden. Do not present a `low` note or `?` line as certain.
- No audio leaves the machine. No online lyric lookup unless the user asks.
- If `whisper-cli`, `ffmpeg` or the venv is missing, say which and give the install line (`brew install whisper-cpp ffmpeg`; `uv sync` in the repo).

## Common mistakes
- Trusting the first tempo (librosa doubles/triples slow shuffles). The pipeline corrects it; if the page shows 160 BPM for a slow blues, run `harmony --force` and read the bass root intervals.
- Reading a bar full of the same low note as a riff. It is bleed; the note JSON says `reason: bleed`.
```

- [ ] **Step 3: README.md**

```markdown
# music-transcribe

Letra, cifra com diagramas e tab com articulações de uma música, a partir do áudio, tudo local.

## Requisitos
- macOS Apple Silicon (MPS) ou CPU; `brew install ffmpeg whisper-cpp uv`
- Python 3.11 (basic-pitch)

## Uso
```bash
uv sync
uv run music-transcribe run ~/Music/song.flac --yes
open ~/Music/song.transcribe/render/cifra.html
```
Etapas: `tags` `stems` `lyrics` `harmony` `notes` `render`, cada uma um subcomando; resultados ficam em `<audio>.transcribe/` e são reaproveitados (`--force` refaz).

## Plugin Claude Code
`/plugin marketplace add jpfaria/music-transcribe` → skill `transcribing-a-song`.

## Confiança
Toda nota, acorde e linha de letra traz confiança (high/medium/low + motivo). A página nunca esconde isso.

## Docs
Spec em `docs/superpowers/specs/`, plano em `docs/superpowers/plans/`.
```

- [ ] **Step 4: Commit, push**

```bash
git add -A && git commit -m "Add the transcribing-a-song skill and README" && git push
```

---

### Task 17: Synthetic end-to-end fixture and slow integration test

**Files:**
- Create: `tests/test_integration.py`, `tests/make_fixture.py`

**Interfaces:** consumes the CLI.

- [ ] **Step 1: Fixture generator (not committed audio; generated at test time)**

`tests/make_fixture.py`:
```python
"""Synthesize a 30 s 'song' in Bbm at 60 BPM 4/4: bass roots Bb-Db-Eb-Bb per bar, chord pad, a lead line with a bend."""
import numpy as np
from conftest import sine_note, midi_to_hz, SR


def make(path):
    bar = 4.0
    roots = [34, 37, 39, 34]                       # Bb1 Db2 Eb2 Bb1
    tri = {34: [58, 61, 65], 37: [61, 65, 68], 39: [63, 66, 70]}
    y = np.zeros(int(8 * bar * SR) + SR, dtype=np.float32)
    lead = [70, 73, 75, 77, 75, 73, 70, 68]
    for b in range(8):
        r = roots[b % 4]; t0 = int(b * bar * SR)
        for k in range(4):
            seg = sine_note(midi_to_hz(r), 0.9, SR, amp=0.4, harmonics=2)
            y[t0 + k * SR: t0 + k * SR + len(seg)] += seg
        for p in tri[r]:
            seg = sine_note(midi_to_hz(p), bar, SR, amp=0.12)
            y[t0: t0 + len(seg)] += seg
        seg = sine_note(midi_to_hz(lead[b]), 1.5, SR, amp=0.35)
        y[t0: t0 + len(seg)] += seg
    import soundfile as sf
    sf.write(str(path), y / (np.abs(y).max() + 1e-6) * 0.9, SR)
    return path
```

- [ ] **Step 2: Slow test**

`tests/test_integration.py`:
```python
import json, shutil, subprocess, sys
import pytest
from pathlib import Path
from make_fixture import make

pytestmark = pytest.mark.slow


@pytest.mark.skipif(shutil.which("whisper-cli") is None or shutil.which("ffmpeg") is None, reason="needs whisper-cli and ffmpeg")
def test_full_pipeline_on_synthetic_song(tmp_path):
    audio = make(tmp_path / "synth.wav")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "music_transcribe.cli", "run", str(audio), "--out", str(out), "--yes",
                        "--model", "small.en", "--instruments", "guitar,bass", "--device", "cpu"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    h = json.loads((out / "harmony" / "harmony.json").read_text())
    assert 55 <= h["bpm"] <= 65
    assert h["key"] in ("Bbm", "Db")
    names = [c["name"] for c in h["chords"]]
    assert names[:4] == ["Bbm", "Db", "Ebm", "Bbm"] or h["loop"] == ["Bbm", "Db", "Ebm", "Bbm"]
    assert (out / "render" / "cifra.html").exists()
    assert json.loads((out / "lyrics" / "lyrics.json").read_text()) == []   # instrumental
```

Add to `cli.py` bottom: `if __name__ == "__main__": app()` so `python -m music_transcribe.cli` works.

- [ ] **Step 3: Run the slow test once manually and record the outcome in `docs/pending.md`**

Run: `uv run pytest -m slow -v -s`
Expected: PASS. If demucs on a synthetic mix puts the pad in `other` instead of `guitar`, the harmony assertion still holds (chroma sums harmonic stems) and the tab may be empty; that is acceptable for the fixture and must be noted in `docs/pending.md`.

- [ ] **Step 4: Commit, push**

```bash
git add -A && git commit -m "Add synthetic end-to-end fixture and slow integration test" && git push
```

---

## Self-review notes

- Spec coverage: tags (T3), stems (T5), lyrics with confidence + fallback model (T4, T6), tempo/meter/grid/chords/inversions/loop/key (T7, T8), notes + bend/vibrato/slide/hammer/pull + confidence + bleed (T9–T11), fretboard for guitar and bass, piano as notes (T11, T14), diagrams with fallback (T12), HTML/txt/MIDI (T13–T14), CLI with cache and subcommands (T15), skill (T16), synthetic integration (T17).
- Type consistency: `Note.articulation` strings are defined in T2 and consumed identically in T10 (`classify_contour`, `detect_legato`) and T13 (`token`). `Harmony.bar0/bar_len/meter` names used in T13/T14 match T2. `OutDir.stage/done/mark_done` used in T15 match T2. `TEMPLATES/SUFFIX` from T8 are reused by T12.
- Review Focus tests live in: T3 (`test_load_mono_resamples_stereo_48k`), T6 + T14 (instrumental), T7 (`test_shuffle_52_is_not_reported_as_156`), T9 (`test_mark_bleed_runs_finds_triplet_drone`), T12 (parametrized voicing test incl. `Ebm7b5`, `Db/F`).
